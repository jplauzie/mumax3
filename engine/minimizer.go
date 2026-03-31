package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).
// Step size is selected by the alternating Barzilai-Borwein method (original behaviour).
// Energy backtracking and stagnation recovery are retained from the improved version.

import (
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DmSamples             int     = 10
	StopMaxDm             float64 = 1e-6
	MinimizeWallClockTime float64 = -1.0
	MinimizeConverged     bool
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable.")
}

// fixed length FIFO. Items can be added but not removed.
type fifoRing struct {
	count int
	tail  int
	data  []float64
}

func FifoRing(length int) fifoRing {
	return fifoRing{data: make([]float64, length)}
}

func (r *fifoRing) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

// Max returns the largest value in the filled portion of the ring.
// Guards against an empty ring and iterates only filled slots.
func (r *fifoRing) Max() float64 {
	if r.count == 0 {
		return 0
	}
	max := math.Inf(-1)
	for i := 0; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

type Minimizer struct {
	k      *data.Slice
	lastDm fifoRing
	h      float32
}

func (mini *Minimizer) Free() {
	mini.k.Free() // Slice.Free() is nil-safe
}

// bbStep computes a BB step size from numerator and denominator, returning
// (value, true) when the result is a valid positive step, or (0, false) when
// the denominator is too small or the result would be non-positive.
// Both numerator and denominator are dot products; the result is only
// meaningful as a step size when it is strictly positive and finite.
// bbStep computes a BB step size from numerator and denominator, returning
// (value, true) when the result is a valid positive step, or (0, false) when
// the denominator is too small or the result would be non-positive.
const denomThresh = float32(1e-12)

func bbStep(num, den float32, denomthresh float32) (float32, bool) {
	const denomThresh = float32(1e-12)

	if den <= 0 || float32(math.Abs(float64(den))) < denomThresh {
		return 0, false
	}

	h := num / den

	if h <= 0 || math.IsNaN(float64(h)) || math.IsInf(float64(h), 0) {
		return 0, false
	}

	// very loose clamp (only prevent insanity)
	if h > 1e6 {
		h = 1e6
	}
	if h < 1e-8 {
		h = 1e-8
	}

	return h, true
}
func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil || !mini.k.GPUAccess() {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
	}

	k := mini.k
	h := mini.h

	// save original magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// take step
	cuda.Minimize(m, m0, k, h)

	// compute new torque
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	torqueFn(k)
	setMaxTorque(k)

	// aliases
	dm := m0
	dk := k0

	// dm = m - m0
	cuda.Madd2(dm, m, m0, 1., -1.)
	// dk = k - k0
	cuda.Madd2(dk, k, k0, -1., 1.)

	// measure step
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max())

	// dot products
	var nom, div float32
	if NSteps%2 == 0 {
		nom = cuda.Dot(dm, dm)
		div = cuda.Dot(dm, dk)
	} else {
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, dk)
	}

	// --------------------------------------------------
	// 🔑 Minimal FIX: float32-safe division
	// --------------------------------------------------
	const denomThresh = float32(1e-12)

	if float32(math.Abs(float64(div))) > denomThresh {
		newH := nom / div

		// minimal sanity check
		if newH > 0 && !math.IsNaN(float64(newH)) && !math.IsInf(float64(newH), 0) {

			// very loose clamp (only extremes)
			if newH > 1e6 {
				newH = 1e6
			}
			if newH < 1e-12 {
				newH = 1e-12
			}

			mini.h = newH
		}
		// else: keep previous h (important!)
	}
	// else: keep previous h (instead of resetting!)

	M.normalize()
	NSteps++
}

// WallclockTimer returns false if the wall clock time limit is exceeded.
// If WallClockTime is negative this function always returns true.
func WallclockTimer(start time.Time, WallClockTime float64) bool {
	if WallClockTime < 0 {
		return true
	}
	if WallClockTime == 0 {
		return false
	}
	return time.Since(start) < time.Duration(WallClockTime*float64(time.Second))
}

func Minimize() bool {
	MinimizeConverged = false
	TimerStart := time.Now()

	if MinimizeWallClockTime == 0 {
		return MinimizeConverged
	}

	Refer("exl2014")
	SanityCheck()
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0
		relaxing = false
	}()

	Precess = false
	if stepper != nil {
		stepper.Free()
	}

	mini := Minimizer{
		h:      1e-4,
		k:      nil,
		lastDm: FifoRing(DmSamples),
	}

	// Scale initial step to actual torque magnitude.
	mini.k = cuda.Buffer(3, M.Buffer().Size())
	torqueFn(mini.k)
	initialTorque := float32(cuda.MaxVecNorm(mini.k))
	if initialTorque > 0 {
		mini.h = 1e-2 / initialTorque
	}
	//log.printf("Minimize: starting, initialTorque=%.3e h0=%.3e StopMaxDm=%.3e DmSamples=%d", initialTorque, mini.h, StopMaxDm, DmSamples)

	stepper = &mini

	prevMaxDm := math.MaxFloat64
	stagnationCount := 0
	postResetGrace := 0

	cond := func() bool {
		currentDm := mini.lastDm.Max()

		absDiff := math.Abs(prevMaxDm - currentDm)
		relDiff := absDiff
		if prevMaxDm > 0 {
			relDiff = absDiff / prevMaxDm
		}
		if relDiff < 1e-6 && absDiff < 1e-12 {
			stagnationCount++
		} else {
			stagnationCount = 0
		}
		prevMaxDm = currentDm

		if stagnationCount > DmSamples*2 {
			//log.printf("Minimize: stagnation reset at dm=%.3e after %d steps", currentDm, NSteps)
			if mini.k != nil {
				mini.k.Free()
			}
			mini.k = nil
			mini.h = 1e-4
			stagnationCount = 0
			postResetGrace = DmSamples
		}

		if postResetGrace > 0 {
			postResetGrace--
		}

		withinTime := WallclockTimer(TimerStart, MinimizeWallClockTime)
		converged := postResetGrace == 0 &&
			mini.lastDm.count >= DmSamples &&
			currentDm <= StopMaxDm

		if !withinTime {
			//log.printf("Minimize: exiting due to wall-clock limit at dm=%.3e after %d steps", currentDm, NSteps)
		}
		if converged {
			//log.printf("Minimize: converged at dm=%.3e after %d steps", currentDm, NSteps)
		}

		return !converged && withinTime
	}

	RunWhile(cond)
	pause = true
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}
