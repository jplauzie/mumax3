package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	DmSamples              int     = 10   // number of dm to keep for convergence check
	StopMaxDm              float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
	MinimizeWallClockTime  float64 = -1.0 // wall-clock time limit for minimization
	MinimizeConverged      bool           // true if minimize converged, and false if the maximum wall-clock time is reached
	MinimizeLineSearchMode int     = 0    // 0: original, 1: cold-start line search only, 2: + bad-step recovery and convergence rescue, 3: + bad-step recovery and GLL energy test every step
	MinimizeMaxStepAngle   float64 = 0    // <=0 disables the angle cap in SD line searches
	MinimizeMaxStall       int     = 3    // consecutive no-progress rescues before rescues are switched off
	MinimizeLogFile        string         // diagnostic: if set, Minimize() appends step, <mz>, E, max|torque| per step to this file
	minLogF                *os.File
	minLog                 *bufio.Writer
)

var dbgStep int // temporary debug counter

// sdFallbackDisplacements are max per-cell |Δm| tried in order (as d/maxDirNorm along k)
// when a line search fails. The first one that passes the GLL test is accepted; if none
// pass, recoverStep reports that no step was taken.
var sdFallbackDisplacements = [...]float64{0.1, 0.05, 0.01}

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check. In line-search mode 3 this is also the number of recent energies used for the non-monotone (GLL) test.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable. An interrupted minimization does not guarantee a correct solution.")
	DeclVar("MinimizeLineSearchMode", &MinimizeLineSearchMode,
		"Minimize() line-search mode. 0: original BB steps, no line search (default). 1: line search for the first step only. 2: as 1, and also recover from bad BB steps (NaN/Inf/0), and try a line search when the dm criterion would declare convergence, continuing if it finds real progress. 3: as 1, and also recover from bad BB steps, and apply the non-monotone GLL energy test (over the last MinimizerSamples energies) to every BB step (Exl et al. 2014; one extra energy evaluation per step).")
	DeclVar("MinimizeMaxStepAngle", &MinimizeMaxStepAngle, "Max rotation angle (degrees) per line-search trial step in Minimize(); <=0 disables (default: 0). Clamped to 89 max.")
	DeclVar("MinimizeMaxStall", &MinimizeMaxStall, "Minimize: number of consecutive no-progress rescues before further rescues are disabled. In mode 3 these are GLL line-search rescues (BB steps are then accepted unconditionally until the energy drops below the level at the start of the streak). In mode 2 these are convergence rescues (the run then exits as converged). Default: 3.")
	DeclVar("MinimizeLogFile", &MinimizeLogFile, "Diagnostic: write step, <mz>, energy and max torque for every Minimize() step to this file. Empty (default) disables. Costs one extra energy evaluation per step.")
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
	mode       int     // line-search mode, captured once at the start of Minimize()
	firstStep  bool    // true only before τ0 has been computed
	stallCount int     // consecutive GLL rescues without progress
	fRef       float64 // energy at the start of the current no-progress streak
	gllOff     bool    // true: GLL rescues disabled until progress is made
	hBadNext   bool    // BB formula degenerate (div == 0); force recovery on the next step (modes 2, 3)
	stuck      bool    // recoverStep found no descent step: M is at a float32-resolution minimum

	lastF float64 // energy at the current M; valid only if haveF
	haveF bool

	rescueStall int     // consecutive convergence candidates without energy progress since the last rescue (mode 2)
	fRescue     float64 // energy at the last accepted convergence-rescue point (mode 2)
	haveRescue  bool
	lastScale   float64 // energy noise scale at the current M; valid only if haveF
}

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
		mini.firstStep = true
		mini.lastEnergy = FifoRing(DmSamples)
		mini.haveF = false
	}

	k := mini.k
	h := mini.h

	// Modes 2 and 3: sanitize the BB step size. No field evaluation is needed.
	//   - NaN, Inf, 0, or a degenerate BB formula on the previous step: bad step, goes to recoverStep.
	//   - hMax keeps dt*dt finite in float32 inside minimize.cu.
	//   - Upper clamp: the Cayley update is an exact rotation by 2*atan(d/2), d = |h|*max|k|.
	//     Once d > 1/f32eps the tangent part of m_new (~4/d) is below float32 resolution of
	//     |m| = 1, so d is clamped there (sign and direction of h are kept). max|k| is only
	//     computed (one reduction) when |h| > hCheck, since the clamp can only fire for large
	//     |h| unless max|k| is unphysically large (> 1/(f32eps*hCheck), about 8e3 T).
	const hMax = 1e18
	const hCheck = 1e3
	badH := false
	if mini.mode >= 2 { // modes 2 and 3
		hf := float64(h)
		if mini.hBadNext || math.IsNaN(hf) || math.IsInf(hf, 0) || hf == 0 {
			badH = true
		} else {
			if math.Abs(hf) > hMax {
				hf = math.Copysign(hMax, hf)
			}
			if math.Abs(hf) > hCheck {
				if kMax := float64(cuda.MaxVecNorm(k)); kMax > 0 && math.Abs(hf)*kMax > 1/f32eps {
					hf = math.Copysign(1/(f32eps*kMax), hf)
				}
			}
			h = float32(hf)
		}
	}
	mini.hBadNext = false

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)

	var trialF float64
	moved := true

	if mini.mode == 0 {
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
		kTrial := cuda.Buffer(3, size)
		defer cuda.Recycle(kTrial)

		energyAtM0 := func() float64 { // cached, or computed lazily
			if !mini.haveF {
				mini.lastF, mini.haveF = energyAtAccepted(), true
				mini.lastScale = lastEnergyScale
			}
			return mini.lastF
		}

		switch {
		case mini.firstStep || badH:
			// Cold start and bad BB step: line search from the cold seed,
			// then fixed-displacement fallback.
			f0 := energyAtM0()
			trialF, moved = mini.recoverStep(m0, f0, mini.lastScale, k, kTrial)
			mini.firstStep = false
			if moved {
				mini.lastF, mini.lastScale, mini.haveF = trialF, lastEnergyScale, true
			}

		case mini.mode == 3:
			f0 := energyAtM0()
			f0Scale := mini.lastScale
			cuda.Minimize(m, m0, k, h)
			trialF = evalEnergyGradient_SD(kTrial)
			switch {
			case math.IsNaN(trialF) || math.IsInf(trialF, 0):
				// BB step produced a non-finite energy: reject it and recover from m0.
				data.Copy(M.Buffer(), m0)
				trialF, moved = mini.recoverStep(m0, f0, f0Scale, k, kTrial)
			case !mini.gllOff && mini.lastEnergy.count > 0 &&
				trialF > mini.lastEnergy.Max()+4*f32eps*lastEnergyScale:
				// GLL rejection (Exl et al. 2014).
				trialF, moved = mini.gllRescue(m0, f0, f0Scale, k, kTrial, h)
			default:
				cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0) // accept BB step; k = torque
			}
			if moved {
				mini.lastF, mini.lastScale, mini.haveF = trialF, lastEnergyScale, true
			}

		default: // modes 1 and 2 after the first step: plain BB, no energy evaluation
			cuda.Minimize(m, m0, k, h)
			torqueFn(k)
			mini.haveF = false
		}

		if mini.mode == 3 && moved {
			fmt.Printf("DBG stall: f=%.15e fRef=%.15e stallCount=%d gllOff=%v\n",
				trialF, mini.fRef, mini.stallCount, mini.gllOff)
			// end a no-progress streak (and re-enable GLL rescues) once the energy has
			// dropped below the streak's reference by more than float32 noise
			if mini.stallCount > 0 && trialF < mini.fRef-4*f32eps*mini.lastScale {
				mini.stallCount = 0
				mini.gllOff = false
			}
			mini.lastEnergy.Add(trialF)
		}
	}

	if !moved {
		// No descent step exists at float32 resolution: M == m0, k unchanged.
		// Tell Minimize() to stop; the dm ring and LastErr are left untouched.
		fmt.Printf("WARNING: Minimize: no descent step found, max|k|=%e; declaring converged\n", cuda.MaxVecNorm(k))
		mini.stuck = true
		setMaxTorque(k) // still report the torque to the user
		NSteps++
		return
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
	} else {
		mini.h = 1e-4
		if mini.mode >= 2 { // modes 2 and 3
			mini.hBadNext = true // let recoverStep handle it
		}
	}

	M.normalize()

	// Mode 2: the dm ring says "converged"; try one line search before exiting.
	if mini.mode == 2 && mini.lastDm.count >= DmSamples && mini.lastDm.Max() <= StopMaxDm {
		mini.convergenceRescue()
	}
	minimizeLogStep(mini.k)
	NSteps++
}

// convergenceRescue is called (mode 2) when the dm ring says "converged". It tries one line
// search along the current torque. If that finds a real energy decrease with a
// displacement larger than StopMaxDm, the BB iteration had stalled: take the step,
// clear the dm ring and carry on. Otherwise leave the ring full so the run stops.
// On return M and mini.k are consistent (k = torque at M) in every case.
func (mini *Minimizer) convergenceRescue() {
	m := M.Buffer()
	size := m.Size()
	k := mini.k

	f0 := energyAtAccepted()
	f0Scale := lastEnergyScale
	noise := 4 * f32eps * f0Scale

	// Did the BB phase since the last rescue make energy progress?
	if mini.haveRescue {
		if f0 < mini.fRescue-noise {
			mini.rescueStall = 0
		} else {
			mini.rescueStall++
		}
		if mini.rescueStall >= MinimizeMaxStall {
			return // bottomed out: let the run converge
		}
	}

	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	dirNorm := float64(cuda.MaxVecNorm(k)) // before the search overwrites k
	newF, step, ok := mini.SD_linesearch(m0, f0, f0Scale, k, 0)
	if !ok {
		return // M, k untouched: converged
	}
	setMaxTorque(k)
	fmt.Printf("DBG convRescue: f0=%e newF=%e disp=%e stop=%e rescueStall=%d\n",
		f0, newF, step*dirNorm, StopMaxDm, mini.rescueStall)

	if newF < f0-noise && step*dirNorm > StopMaxDm {
		// real progress available: BB had stalled
		mini.fRescue, mini.haveRescue = newF, true
		mini.lastDm = FifoRing(DmSamples) // re-arm the convergence check
		mini.h = float32(step)            // restart BB from the step the search accepted
		mini.haveF = false
	}
	// else: the step was accepted (it did not raise E) but there is nothing
	// significant left, so the ring stays full and the run exits.
}

// gllRescue handles a GLL-rejected BB step: it runs a line search from m0 seeded at |h|.
// If the search fails, it falls back to recoverStep (cold-seed search, then fixed
// displacements along +k) rather than taking the rejected BB step.
// It tracks no-progress streaks: after MinimizeMaxStall rescues in a row
// without the energy dropping below the streak's reference, GLL rescues are disabled
// (mini.gllOff) from the next step on. Returns the energy at the new M and whether a
// step was taken; k is the torque at M.
func (mini *Minimizer) gllRescue(m0 *data.Slice, f0, f0Scale float64, k, kTrial *data.Slice, h float32) (float64, bool) {
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
	newF, _, ok := mini.SD_linesearch(m0, f0, f0Scale, k, math.Abs(float64(h)))
	fmt.Printf("DBG GLL result: ok=%v newF=%e (f0=%e)\n", ok, newF, f0)
	if ok {
		return newF, true
	}

	// Search failed: M and k are back at m0. Fall back to recoverStep instead of
	// taking the rejected BB step.
	return mini.recoverStep(m0, f0, f0Scale, k, kTrial)
}

// recoverStep produces a step from m0 when there is no usable BB step (cold start, or a
// non-finite BB step/energy, or a GLL rescue whose seeded search failed): a line search
// from the cold seed, then fixed displacements from sdFallbackDisplacements (the first
// that passes the GLL test is accepted). Also called from gllRescue. Returns the energy
// at the new M and true; k is the torque there. If no step is found (or the torque is
// zero), M is left at m0, k is unchanged, and it returns (f0, false).
func (mini *Minimizer) recoverStep(m0 *data.Slice, f0, f0Scale float64, k, kTrial *data.Slice) (float64, bool) {
	data.Copy(M.Buffer(), m0)
	if newF, _, ok := mini.SD_linesearch(m0, f0, f0Scale, k, 0); ok {
		return newF, true
	}

	maxDirNorm := float64(cuda.MaxVecNorm(k))
	if !(maxDirNorm > 0) { // zero (or NaN) torque: nothing to step along
		return f0, false
	}
	ref := f0
	if mini.lastEnergy.count > 0 {
		ref = mini.lastEnergy.Max()
	}
	for _, d := range sdFallbackDisplacements {
		cuda.Minimize(M.Buffer(), m0, k, float32(d/maxDirNorm))
		f := evalEnergyGradient_SD(kTrial)
		if f <= ref+4*f32eps*lastEnergyScale { // false for NaN
			cuda.Madd2(k, kTrial, kTrial, -1.0, 0.0)
			return f, true
		}
	}
	// No descent step found at any scale: M back at m0, k still the torque at m0.
	data.Copy(M.Buffer(), m0)
	return f0, false
}

// SD_linesearch runs an inexact line search from m0 (energy f0) along
// the fixed direction k (m0's torque), used for τ0, for bad-BB-step recovery,
// for BB-step rejections and for convergence rescues.
//
// Contract:
//   - ok == true:  M is at the accepted point, k holds the torque there,
//     newF is the energy there and step is the accepted step.
//   - ok == false: M == m0, k is unchanged (still the torque at m0),
//     newF == f0 and step == 0. Callers may rely on this.
func (mini *Minimizer) SD_linesearch(m0 *data.Slice, f0, f0Scale float64, k *data.Slice, seed float64) (newF, step float64, ok bool) {
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
	fmt.Printf("DBG SD_linesearch: max|k|=%e seed=%e step0=%e f0=%e\n", maxDirNorm, seed, step0, f0)

	// k is the search direction s: MTlinesearch only reads it.
	f, stp, info := MTlinesearch(m0, f0, gradient, maxDirNorm, step0, k, evalEnergyGradient_SD, 0, MinimizeMaxStepAngle, f0Scale)

	ok = stp > 0 && !math.IsNaN(f) && !math.IsInf(f, 0) &&
		(info == lsConverged || info == lsAtMaxStep || f < f0)
	fmt.Printf("DBG SD_linesearch result: ok=%v f=%e step=%e info=%d evals-done\n", ok, f, stp, info)
	if !ok {
		// M may be at a trial point (step > 0 with lsAtMinStep / lsNoProgress /
		// lsMaxEvals and f >= f0, or a non-finite trial). Put it back and leave k alone.
		data.Copy(M.Buffer(), m0)
		return f0, 0, false
	}

	cuda.Madd2(k, gradient, gradient, -1.0, 0.0) // k = torque at the accepted point
	return f, stp, true
}

func (mini *Minimizer) Free() {
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

	mode := MinimizeLineSearchMode
	if mode < 0 || mode > 3 {
		util.Fatalf("MinimizeLineSearchMode must be 0..3, got %d", mode)
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
		mode:   mode,
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	cond := func() bool {
		return !mini.stuck &&
			(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) &&
			WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	RunWhile(cond)
	closeMinimizeLog()
	pause = true
	// if the loop ended because of convergence, then MinimizeConverged is true. If the loop ended because of wall-clock time, then MinimizeConverged is false.
	MinimizeConverged = mini.stuck || !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}

// lastEnergyScale is max(|E|, Σ|E_i|) from the most recent minimizerEnergy() call.
// It is the scale for float32 noise in E: cancellation between terms makes |E| alone
// too small. It does not capture cancellation inside a single term's reduction.
var lastEnergyScale float64

// minimizerEnergy sums energyTerms like GetTotalEnergy, but does not panic on NaN, so
// callers can reject a bad trial point. It also records lastEnergyScale.
func minimizerEnergy() float64 {
	E, sum := 0., 0.
	for _, f := range energyTerms {
		e := f()
		E += e
		sum += math.Abs(e)
	}
	lastEnergyScale = math.Max(math.Abs(E), sum)
	return E
}

// energyAtAccepted is minimizerEnergy for the current (accepted) M. A non-finite energy
// there cannot be rolled back, so this fails loudly like GetTotalEnergy does.
func energyAtAccepted() float64 {
	E := minimizerEnergy()
	if math.IsNaN(E) || math.IsInf(E, 0) {
		panic("Minimize: non-finite energy at the accepted magnetization")
	}
	return E
}

func evalEnergyGradient_SD(g *data.Slice) float64 {
	M.normalize()
	torqueFn(g)
	cuda.Madd2(g, g, g, -1.0, 0.0) // g = -torque = positive gradient
	return minimizerEnergy()
}

// minimizeLogStep writes one diagnostic line. Uses GetTotalEnergy (not minimizerEnergy)
// so it does not touch lastEnergyScale.
func minimizeLogStep(k *data.Slice) {
	if MinimizeLogFile == "" {
		return
	}
	if minLog == nil {
		f, err := os.Create(OD() + MinimizeLogFile)
		if err != nil {
			panic(err)
		}
		minLogF, minLog = f, bufio.NewWriter(f)
		fmt.Fprintln(minLog, "# step mz energy max_torque")
	}
	mz := sAverageMagnet(M.Buffer())[2]
	fmt.Fprintf(minLog, "%d %.12e %.17e %.9e\n", NSteps, mz, GetTotalEnergy(), cuda.MaxVecNorm(k))
	if NSteps%1000 == 0 {
		minLog.Flush()
	}
}

func closeMinimizeLog() {
	if minLog != nil {
		minLog.Flush()
		minLogF.Close()
		minLog, minLogF = nil, nil
	}
}
