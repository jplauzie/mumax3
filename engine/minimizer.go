package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).
// Step size is selected by the alternating Barzilai-Borwein method (original behaviour).
// Energy backtracking and stagnation recovery are retained from the improved version.

import (
	"log"
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

func bbStep(num, den float32, denomThresh float32) (float32, bool) {
	// Reject tiny denominators (avoids blow-ups)
	if float32(math.Abs(float64(den))) <= denomThresh {
		return 0, false
	}

	h := num / den

	// Reject invalid values
	if h <= 0 || math.IsNaN(float64(h)) || math.IsInf(float64(h), 0) {
		return 0, false
	}

	// Clamp to safe range
	const hMin = float32(1e-8)
	const hMax = float32(1e8)

	if h < hMin {
		h = hMin
	}
	if h > hMax {
		h = hMax
	}

	return h, true
}
func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	// GPUAccess() check catches post-Free() disabled slices, which have a
	// non-nil pointer but zeroed memType and would panic on DevPtr().
	if mini.k == nil || !mini.k.GPUAccess() {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
	}

	k := mini.k

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	const maxRetries = 8
	const backtrackFactor = float32(0.5)
	// backtrackThreshold: only run energy backtracking when h is large
	// enough that an explosive step is plausible. Below this value the
	// step is taken unconditionally, matching the original method.
	const backtrackThreshold = float32(1e-3)

	accepted := false
	if mini.h > backtrackThreshold {
		mTrial := cuda.Buffer(3, size)
		defer cuda.Recycle(mTrial)

		E0 := GetTotalEnergy()

		for i := 0; i < maxRetries; i++ {
			data.Copy(mTrial, m0)
			cuda.Minimize(mTrial, m0, k, mini.h)
			data.Copy(m, mTrial)
			M.normalize()
			if GetTotalEnergy() <= E0 {
				accepted = true
				break
			}
			data.Copy(m, m0)
			mini.h *= backtrackFactor
		}

		if accepted && mini.h < backtrackThreshold {
			// Backtracking shrank h below the useful range. Reset to a safe
			// value so BB sees a real displacement on the next step.
			log.Printf("Minimize: backtrack shrank h to %.3e at step=%d, resetting to 1e-4", mini.h, NSteps)
			mini.h = 1e-4
		}
	} else {
		// h is small — take unconditionally, matching original behaviour.
		cuda.Minimize(m, m0, k, mini.h)
		M.normalize()
		accepted = true
	}

	if !accepted {
		log.Printf("Minimize: all retries failed at step=%d dm=%.3e h=%.3e, resetting to 1e-4", NSteps, mini.lastDm.Max(), mini.h)
		data.Copy(m, m0)
		mini.h = 1e-4
		NSteps++
		return
	}

	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	torqueFn(k)
	setMaxTorque(k)

	dm := m0
	dk := k0

	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.)

	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max())

	dotDmDm := cuda.Dot(dm, dm)
	dotDmDk := cuda.Dot(dm, dk)
	dotDkDk := cuda.Dot(dk, dk)

	log.Printf("Minimize: step=%d dm=%.3e dotDmDm=%.3e dotDmDk=%.3e dotDkDk=%.3e h=%.3e",
		NSteps, max_dm, dotDmDm, dotDmDk, dotDkDk, mini.h)

	// Alternating BB step selection.
	// BB1 (even steps, long step):  h = dot(dm,dm) / dot(dm,dk)
	//   Valid only when dotDmDk > 0; a negative dotDmDk means the gradient
	//   reversed direction and BB1 would yield a negative (invalid) h.
	// BB2 (odd steps, short step):  h = dot(dm,dk) / dot(dk,dk)
	//   dotDkDk is a squared norm so non-negative, but dotDmDk can be
	//   negative, making this ratio negative too — also invalid.
	// bbStep() checks both the denominator magnitude and the sign of the
	// result, so neither formula can produce a zero or negative step size.
	// When the preferred formula is invalid we try the other; when both
	// are invalid we leave h unchanged and log.

	var newH float32
	computed := false
	var reason string

	if NSteps%2 == 0 {
		// Prefer BB1 on even steps.
		if h, ok := bbStep(dotDmDm, dotDmDk, denomThresh); ok {
			newH = h
			computed = true
		} else if h, ok := bbStep(dotDmDk, dotDkDk, denomThresh); ok {
			newH = h
			computed = true
			reason = "BB1 invalid, used BB2 fallback"
		}
	} else {
		// Prefer BB2 on odd steps.
		if h, ok := bbStep(dotDmDk, dotDkDk, denomThresh); ok {
			newH = h
			computed = true
		} else if h, ok := bbStep(dotDmDm, dotDmDk, denomThresh); ok {
			newH = h
			computed = true
			reason = "BB2 invalid, used BB1 fallback"
		}
	}

	if computed {
		if reason != "" {
			log.Printf("Minimize: step=%d %s h=%.3e", NSteps, reason, newH)
		}
		mini.h = newH
	} else {
		// Both BB formulas invalid this step (gradient reversal or near-zero
		// displacement). Keep h at its current value — it is already at a
		// safe scale and the next step will re-establish a valid estimate.
		log.Printf("Minimize: step=%d both BB invalid (dotDmDk=%.3e dotDkDk=%.3e), h unchanged=%.3e",
			NSteps, dotDmDk, dotDkDk, mini.h)
	}

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
	log.Printf("Minimize: starting, initialTorque=%.3e h0=%.3e StopMaxDm=%.3e DmSamples=%d",
		initialTorque, mini.h, StopMaxDm, DmSamples)

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
			log.Printf("Minimize: stagnation reset at dm=%.3e after %d steps", currentDm, NSteps)
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
			log.Printf("Minimize: exiting due to wall-clock limit at dm=%.3e after %d steps", currentDm, NSteps)
		}
		if converged {
			log.Printf("Minimize: converged at dm=%.3e after %d steps", currentDm, NSteps)
		}

		return !converged && withinTime
	}

	RunWhile(cond)
	pause = true
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}
