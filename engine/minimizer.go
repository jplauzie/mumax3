package engine

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
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize.")
}

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

// CHANGE 5: guard empty ring; only iterate actually-filled slots.
// Original always started from index 0 and read r.count iterations,
// which is correct only after the ring wraps — before wrap, stale
// zeros in unfilled slots could be returned as the max.
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

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

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
	const backtrackThreshold = float32(1e-3)

	accepted := false
	if mini.h > backtrackThreshold {
		// h is large enough that a bad step is plausible — use energy
		// backtracking to guard against explosive moves.
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
	} else {
		// h is small — take the step unconditionally as the original
		// code did. BB update handles convergence from here.
		cuda.Minimize(m, m0, k, mini.h)
		M.normalize()
		accepted = true
	}

	if !accepted {
		data.Copy(m, m0)
		mini.h = 1e-4
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

	const hMin = float32(1e-10)
	const hMax = float32(1.0)
	var nom, div float32
	if NSteps%2 == 0 {
		nom = cuda.Dot(dm, dm)
		div = cuda.Dot(dm, dk)
	} else {
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, dk)
	}
	if div != 0. && nom/div > hMin {
		h := nom / div
		if h > hMax {
			h = hMax
		}
		mini.h = h
	} else {
		mini.h *= backtrackFactor
	}

	NSteps++
}

func (mini *Minimizer) Free() {
	mini.k.Free()
}

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

	// Scale initial step to actual torque magnitude rather than using a
	// fixed 1e-4, which can be orders of magnitude wrong for states far
	// from equilibrium. mini.k is pre-filled here so Step() skips its
	// own init block on the first call.
	mini.k = cuda.Buffer(3, M.Buffer().Size())
	torqueFn(mini.k)
	initialTorque := float32(cuda.MaxVecNorm(mini.k))
	if initialTorque > 0 {
		mini.h = 1e-2 / initialTorque
	}

	stepper = &mini

	prevMaxDm := math.MaxFloat64
	stagnationCount := 0

	cond := func() bool {
		currentDm := mini.lastDm.Max()

		if math.Abs(prevMaxDm-currentDm) < 1e-12 {
			stagnationCount++
		} else {
			stagnationCount = 0
		}
		prevMaxDm = currentDm

		if stagnationCount > DmSamples*2 {
			// Free the current torque buffer and set to nil so that
			// Step()'s nil check reallocates it cleanly on the next call,
			// rather than leaving a post-Free() disabled slice that passes
			// the == nil check but panics on DevPtr().
			if mini.k != nil {
				mini.k.Free()
			}
			mini.k = nil
			mini.h = 1e-4
			stagnationCount = 0
		}

		return (mini.lastDm.count < DmSamples || currentDm > StopMaxDm) &&
			WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	RunWhile(cond)
	pause = true
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}
