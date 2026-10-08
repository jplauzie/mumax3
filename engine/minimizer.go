package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"fmt"
	"math"
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
	MinimizeNonMonotone   bool    = false
	MinimizeMaxStepAngle  float64 = 0 // <=0 disables the angle cap in SD line searches
	MinimizeMaxStall      int     = 3 // consecutive no-progress GLL line searches before GLL rescues are switched off
)

const sdRejectShrink = 1.0 // seed for a GLL-rescue line search = sdRejectShrink*|h|; tune (e.g. 0.25) after measuring
var dbgStep int            // temporary debug counter

// sdFallbackDisplacements are max per-cell |Δm| tried in order (as d/maxDirNorm along k)
// when a line search fails. Each is accepted if it passes the GLL test; the last is unconditional.
var sdFallbackDisplacements = [...]float64{0.1, 0.05, 0.01}

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest descent method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable. An interrupted minimization does not guarantee a correct solution.")
	DeclVar("ExlEnergyWindow", &ExlEnergyWindow, "Number of recent energy values kept for the non-monotone line-search fallback in Minimize() (Exl et al. 2014). Default: 20.")
	DeclVar("MinimizeUseLineSearch", &MinimizeUseLineSearch, "If true, use an inexact line search (Exl et al. 2014) for the initial BB step and for non-monotone-rejected steps. If false, reverts to the original fixed h=1e-4 seed with no line search fallback. Default: false.")
	DeclVar("MinimizeNonMonotone", &MinimizeNonMonotone, "If true and MinimizeUseLineSearch is enabled, BB steps that increase energy beyond the recent ExlEnergyWindow max trigger a line-search fallback (Exl et al. 2014). If false (default), BB steps are always accepted unconditionally after the initial line search, regardless of energy increase.")
	DeclVar("MinimizeMaxStepAngle", &MinimizeMaxStepAngle, "Max rotation angle (degrees) per line-search trial step in Minimize(); <=0 disables (default: 0). Clamped to 89 max.")
	DeclVar("MinimizeMaxStall", &MinimizeMaxStall, "Minimize with MinimizeNonMonotone: after this many consecutive GLL line-search rescues without energy progress, further rescues are disabled (BB steps accepted unconditionally) until the energy drops below the level at the start of the streak. Default: 3.")
}

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
	lastEnergy fifoRing // energy history for non-monotone globalization
	h          float32
	firstStep  bool    // true only before τ0 has been computed
	stallCount int     // consecutive GLL rescues without progress
	fRef       float64 // energy at the start of the current no-progress streak
	gllOff     bool    // true: GLL rescues disabled until progress is made
	hBadNext   bool    // BB formula degenerate (div == 0); force recovery on the next step
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
		if mini.firstStep {
			fmt.Printf("DBG old path first step: max|k|=%e h=%e\n", cuda.MaxVecNorm(k), h)
		}
		cuda.Minimize(m, m0, k, h)
		torqueFn(k)
		trialF = 0 // unused in this path
		mini.firstStep = false
	} else {
		f0 := GetTotalEnergy()

		kTrial := cuda.Buffer(3, size)
		defer cuda.Recycle(kTrial)

		// Bad BB step: NaN, Inf or exactly 0 (a zero step would feed dm=0 into the
		// convergence ring). Negative h is deliberately allowed.
		hBad := mini.hBadNext || math.IsNaN(float64(h)) || math.IsInf(float64(h), 0) || h == 0
		mini.hBadNext = false

		if mini.firstStep || hBad {
			// Trigger 1 (cold start) and trigger 2 (bad BB step): line search from the
			// cold seed, then fixed-displacement fallback.
			trialF = mini.recoverStep(m0, f0, k, kTrial)
			mini.firstStep = false
		} else {
			cuda.Minimize(m, m0, k, h)
			trialF = evalEnergyGradient_SD(kTrial)

			switch {
			case math.IsNaN(trialF) || math.IsInf(trialF, 0):
				// BB step produced a non-finite energy: treat as a bad BB step.
				data.Copy(M.Buffer(), m0)
				trialF = mini.recoverStep(m0, f0, k, kTrial)
			case MinimizeNonMonotone && !mini.gllOff && mini.lastEnergy.count > 0 &&
				trialF > mini.lastEnergy.Max()+4*f32eps*math.Abs(trialF):
				// Trigger 3: GLL rejection (Exl et al. 2014).
				trialF = mini.gllRescue(m0, f0, k, kTrial, h)
			default:
				cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0) // accept BB step; k = torque
			}
		}
		mini.updateStall(trialF)
		mini.lastEnergy.Add(trialF)
	}

	setMaxTorque(k)

	dm := m0
	dk := k0
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.)

	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	if dbgStep < 15 {
		fmt.Printf("DBG step %d: max_dm=%e ring.Max=%e ring.count=%d max|k|=%e h=%e\n",
			dbgStep, max_dm, mini.lastDm.Max(), mini.lastDm.count, cuda.MaxVecNorm(k), h)
		dbgStep++
	}
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
		mini.hBadNext = false
	} else if MinimizeUseLineSearch {
		mini.hBadNext = true // let recoverStep handle it
		mini.h = 1e-4
	} else {
		mini.h = 1e-4 // original behavior
	}

	M.normalize()
	NSteps++
}

// gllRescue handles a GLL-rejected BB step: it runs a line search from m0 seeded at
// sdRejectShrink*|h|. If the search fails, the rejected BB step is taken (original
// behavior). It tracks no-progress streaks: after MinimizeMaxStall rescues in a row
// without the energy dropping below the streak's reference, GLL rescues are disabled
// (mini.gllOff) from the next step on. Returns the energy at the new M; k is the torque there.
func (mini *Minimizer) gllRescue(m0 *data.Slice, f0 float64, k, kTrial *data.Slice, h float32) float64 {
	fmt.Printf("DBG GLL rescue: f0=%e windowMax=%e stallCount=%d gllOff=%v\n",
		f0, mini.lastEnergy.Max(), mini.stallCount, mini.gllOff)
	if mini.stallCount == 0 {
		mini.fRef = f0
	}
	mini.stallCount++
	if mini.stallCount >= MinimizeMaxStall {
		mini.gllOff = true // takes effect next step; this rescue still runs
	}

	data.Copy(M.Buffer(), m0) // early returns of the search leave M untouched
	newF, ok := mini.SD_linesearch(m0, f0, k, sdRejectShrink*math.Abs(float64(h)))
	fmt.Printf("DBG GLL result: ok=%v newF=%e (f0=%e)\n", ok, newF, f0)
	if ok {
		return newF
	}

	// Search failed: M and k are back at m0. Take the BB step that was rejected.
	cuda.Minimize(M.Buffer(), m0, k, h)
	f := evalEnergyGradient_SD(kTrial)
	cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0)
	return f
}

// recoverStep produces a step from m0 when there is no usable BB step (cold start, or a
// non-finite BB step/energy): a line search from the cold seed, then fixed displacements
// from sdFallbackDisplacements (GLL-tested, last one unconditional). Returns the energy at
// the new M; k is the torque there. M is left at m0 only if k is zero (already converged).
func (mini *Minimizer) recoverStep(m0 *data.Slice, f0 float64, k, kTrial *data.Slice) float64 {
	data.Copy(M.Buffer(), m0)
	if newF, ok := mini.SD_linesearch(m0, f0, k, 0); ok {
		return newF
	}

	maxDirNorm := float64(cuda.MaxVecNorm(k))
	if !(maxDirNorm > 0) { // zero (or NaN) torque: nothing to step along
		data.Copy(M.Buffer(), m0)
		return f0
	}
	ref := f0
	if mini.lastEnergy.count > 0 {
		ref = mini.lastEnergy.Max()
	}
	var f float64
	last := len(sdFallbackDisplacements) - 1
	for i, d := range sdFallbackDisplacements {
		cuda.Minimize(M.Buffer(), m0, k, float32(d/maxDirNorm))
		f = evalEnergyGradient_SD(kTrial)
		if i == last || f <= ref+4*f32eps*math.Abs(f) {
			break
		}
	}
	cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0)
	return f
}

// updateStall ends a no-progress streak (and re-enables GLL rescues) once the energy
// has dropped below the streak's reference by more than float32 noise.
func (mini *Minimizer) updateStall(f float64) {
	fmt.Printf("DBG stall: f=%.15e fRef=%.15e stallCount=%d gllOff=%v\n",
		f, mini.fRef, mini.stallCount, mini.gllOff)
	if mini.stallCount > 0 && f < mini.fRef-4*f32eps*math.Abs(mini.fRef) {
		mini.stallCount = 0
		mini.gllOff = false
	}
}

// SD_linesearch runs an inexact line search from m0 (energy f0) along
// the fixed direction k (m0's torque), used for τ0 and for BB-step
// rejections. Restores mini.k to the accepted step's raw torque and
// returns the accepted energy.
func (mini *Minimizer) SD_linesearch(m0 *data.Slice, f0 float64, k *data.Slice, seed float64) (newF float64, ok bool) {
	size := m0.Size()
	gradient := cuda.Buffer(3, size)
	defer cuda.Recycle(gradient)
	cuda.Madd2(gradient, k, k, -1.0, 0.0) // g = -k, positive gradient at m0

	maxDirNorm := float64(cuda.MaxVecNorm(k))
	const coldDisplacement = 0.1 // initial max |Δm| per cell on a cold start; tunable

	step0 := seed
	if !(step0 > 0) || math.IsInf(step0, 0) { // also catches NaN
		if maxDirNorm > 0 {
			step0 = coldDisplacement / maxDirNorm
		}
	}

	// k is the search direction s: MTlinesearch only reads it.
	f, step, info := MTlinesearch(m0, f0, gradient, maxDirNorm, step0, k, evalEnergyGradient_SD, 0, MinimizeMaxStepAngle)

	ok = step > 0 && !math.IsNaN(f) && !math.IsInf(f, 0) &&
		(info == lsConverged || info == lsAtMaxStep || f < f0)

	if !ok {
		// M may be at a trial point (step > 0 with lsAtMinStep / lsNoProgress /
		// lsMaxEvals and f >= f0). Put it back and leave k alone.
		data.Copy(M.Buffer(), m0)
		return f0, false
	}

	cuda.Madd2(k, gradient, gradient, -1.0, 0.0) // k = torque at the accepted point
	return f, true
}

func (mini *Minimizer) Free() {
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

	// if wall-clock time is zero, skip minimization entirely (zero steps), and don't change any settings
	MinimizeConverged = false
	TimerStart := time.Now()
	if MinimizeWallClockTime == 0 {
		MinimizeConverged = false
		return MinimizeConverged
	}

	Refer("exl2014")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	mini := Minimizer{
		h:      1e-4,
		k:      nil,
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) && WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	RunWhile(cond)
	pause = true
	// if the loop ended because of convergence, then MinimizeConverged is true. If the loop ended because of wall-clock time, then MinimizeConverged is false.
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}

func evalEnergyGradient_SD(g *data.Slice) float64 {
	M.normalize()
	torqueFn(g)
	cuda.Madd2(g, g, g, -1.0, 0.0) // g = -torque = positive gradient
	return GetTotalEnergy()
}
