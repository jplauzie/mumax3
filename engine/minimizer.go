package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DmSamples             int     = 10   // number of dm to keep for convergence check
	StopMaxDm             float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
	MinimizeWallClockTime float64 = -1.0 // wall-clock time limit for minimization
	MinimizeConverged     bool           // true if minimize converged, and false if the maximum wall-clock time is reached
	ExlEnergyWindow       int     = 20   // number of recent energies kept for the non-monotone line-search fallback (Exl et al. 2014)
	MinimizeUseLineSearch bool    = false
	MinimizePersist       bool    = false
	MinimizeNonMonotone   bool    = false
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable. An interrupted minimization does not guarantee a correct solution.")
	DeclVar("ExlEnergyWindow", &ExlEnergyWindow, "Number of recent energy values kept for the non-monotone line-search fallback in Minimize() (Exl et al. 2014). Default: 20.")
	DeclVar("MinimizeUseLineSearch", &MinimizeUseLineSearch, "If true, use an inexact line search (Exl et al. 2014) for the initial BB step and for non-monotone-rejected steps. If false, reverts to the original fixed h=1e-4 seed with no line search fallback. Default: false.")
	DeclVar("MinimizePersist", &MinimizePersist, "If true, reuse the Minimizer's BB step size and torque state across Minimize() calls instead of resetting each time. Default: false.")
	DeclVar("MinimizeNonMonotone", &MinimizeNonMonotone, "If true (default) and MinimizeUseLineSearch is enabled, BB steps that increase energy beyond the recent ExlEnergyWindow max trigger a line-search fallback (Exl et al. 2014). If false, BB steps are always accepted unconditionally after the initial line search, regardless of energy increase.")
}

var persistentMinimizer *Minimizer

// fixed length FIFO. Items can be added but not removed
type fifoRing struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
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

func (r *fifoRing) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

type Minimizer struct {
	k          *data.Slice // torque saved to calculate time step
	lastDm     fifoRing
	lastEnergy fifoRing // NEW: energy history for non-monotone globalization
	h          float32
	firstStep  bool // NEW: true only before τ0 has been computed
}

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
		mini.firstStep = true
		mini.lastEnergy = FifoRing(ExlEnergyWindow)
	}

	k := mini.k
	h := mini.h

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)

	var trialF float64

	if !MinimizeUseLineSearch {
		// Original behavior: no line search at all, fixed h seed,
		// unconditional BB step every time.
		cuda.Minimize(m, m0, k, h)
		torqueFn(k)
		trialF = 0 // unused in this path
		mini.firstStep = false
	} else {
		f0 := GetTotalEnergy()

		if mini.firstStep {
			trialF = mini.lineSearchFallback(m0, f0, k, h)
			mini.firstStep = false
		} else {
			cuda.Minimize(m, m0, k, h)

			kTrial := cuda.Buffer(3, size)
			defer cuda.Recycle(kTrial)
			trialF = evalEnergyGradientSteepest(kTrial)

			if MinimizeNonMonotone && mini.lastEnergy.count > 0 && trialF > mini.lastEnergy.Max() {
				data.Copy(M.Buffer(), m0)
				trialF = mini.lineSearchFallback(m0, f0, k, h)
			} else {
				cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0)
			}
		}
		mini.lastEnergy.Add(trialF)
	}

	setMaxTorque(k)

	dm := m0
	dk := k0
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.)

	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max())

	var nom, div float32
	if NSteps%2 == 0 {
		nom = cuda.Dot(dm, dm)
		div = cuda.Dot(dm, dk)
	} else {
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, dk)
	}
	if div != 0. {
		mini.h = nom / div
	} else {
		mini.h = 1e-4
	}

	M.normalize()
	NSteps++
}

// lineSearchFallback runs an inexact line search from wa (energy f0) along
// the fixed direction k (m0's torque), used for τ0 and for BB-step
// rejections. Restores mini.k to the accepted step's raw torque and
// returns the accepted energy.
func (mini *Minimizer) lineSearchFallback(wa *data.Slice, f0 float64, k *data.Slice, h float32) float64 {
	size := wa.Size()
	g := cuda.Buffer(3, size)
	defer cuda.Recycle(g)
	cuda.Madd2(g, k, k, -1.0, 0.0) // g = -k, positive gradient at wa

	var newF, newStp float64
	if LBFGSUseArmijo { // reuse the same toggle, or introduce a dedicated one -- see note
		newF, newStp, _ = armijoSearch(wa, f0, g, float64(h), k, evalEnergyOnlySteepest, evalEnergyGradientSteepest, 0, 0)
	} else {
		newF, newStp, _ = cvsrch(wa, f0, g, float64(h), k, evalEnergyGradientSteepest, 0, 0)
	}
	mini.h = float32(newStp)
	// g now holds the positive gradient at the accepted point; k should
	// hold raw torque, so flip back.
	cuda.Madd2(k, g, g, -1.0, 0.0)
	return newF
}

func (mini *Minimizer) Free() {
	if mini == persistentMinimizer {
		return // survive RunWhile's per-call Free(); real cleanup only via freeBuffers()
	}
	mini.freeBuffers()
}

func (mini *Minimizer) freeBuffers() {
	if mini.k != nil {
		mini.k.Free()
		mini.k = nil
	}
}

// helper function that returns false if the wall clock time limit is exceeded. If the wall-clock time is negative, this function always returns true.
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

	var mini *Minimizer
	if MinimizePersist {
		if persistentMinimizer == nil {
			persistentMinimizer = &Minimizer{h: 1e-4, k: nil, lastDm: FifoRing(DmSamples)}
		}
		mini = persistentMinimizer
	} else {
		mini = &Minimizer{h: 1e-4, k: nil, lastDm: FifoRing(DmSamples)}
	}

	if stepper != nil && stepper != mini {
		stepper.Free()
	}
	stepper = mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) && WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	mini.lastDm = FifoRing(DmSamples)

	RunWhile(cond)
	pause = true
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)

	if mini != persistentMinimizer {
		stepper.Free()
	}
	return MinimizeConverged
}

// evalEnergyGradientSteepest updates the magnetization, normalizes it, and
// writes the positive gradient (-torque) into g, returning total energy.
// Mirrors LBFGSMinimizer.EnergyAndGradient's sign convention so both
// minimizers can share cvsrch/armijoSearch, which expect a positive
// gradient (dginit = dot(g,s) < 0 for a valid descent direction s).
func evalEnergyGradientSteepest(g *data.Slice) float64 {
	M.normalize()
	torqueFn(g)
	cuda.Madd2(g, g, g, -1.0, 0.0) // g = -torque = positive gradient
	return GetTotalEnergy()
}

func evalEnergyOnlySteepest() float64 {
	M.normalize()
	return GetTotalEnergy()
}
