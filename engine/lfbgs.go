package engine

import (
	"fmt"
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Based heavily on "Numerical Optimization" by Nocedal and Wright, 2nd edition, Springer, 2006.

var (
	LBFGSTolerance float64 = 1e-5
	LBFGSMaxIter   int     = 10000
	LBFGSVerbose   int     = 0
	//Nocedal suggests between 3-20
	LBFGSHistory      int     = 5
	LBFGSMaxStepAngle float64 = 85.0 // max degrees any cell's m may rotate in one trial step (<=0 disables)
)

// Create a zero-argument wrapper function for the script to call
func RunMinimizeLBFGS() bool {
	minimizer := NewLBFGSMinimizer(LBFGSTolerance, LBFGSMaxIter, LBFGSVerbose)
	minimizer.History = LBFGSHistory
	minimizer.MaxStepAngle = LBFGSMaxStepAngle
	return minimizer.MinimizeLBFGS()
}

// Register the function and variables with the MuMax3 script parser
func init() {
	DeclFunc("MinimizeLBFGS", RunMinimizeLBFGS, "Use the L-BFGS method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded.")

	DeclVar("LBFGSTolerance", &LBFGSTolerance, "Convergence tolerance for the L-BFGS minimizer (default: 1e-5).")
	DeclVar("LBFGSMaxIter", &LBFGSMaxIter, "Maximum number of iterations for the L-BFGS minimizer (default: 10000).")
	DeclVar("LBFGSVerbose", &LBFGSVerbose, "Verbosity level of the L-BFGS minimizer: 0=silent, 1=basic, 2=detailed (default: 0).")
	DeclVar("LBFGSHistory", &LBFGSHistory, "Number of previous gradients to store for the L-BFGS inverse Hessian approximation (default: 5).")
	DeclVar("LBFGSMaxStepAngle", &LBFGSMaxStepAngle, "Maximum angle (degrees) magnetization may rotate in a single L-BFGS trial step; prevents jumping to an unrelated energy basin. Set <=0 to disable (default: 45).")
}

// LBFGSMinimizer implements the L-BFGS optimization routine, and satisfies
// the Stepper interface so it can be driven through RunWhile like Minimizer.
type LBFGSMinimizer struct {
	Tolerance    float64
	MaxIter      int
	Verbose      int
	History      int // 'm' parameter in L-BFGS (default: 5)
	MaxStepAngle float64

	// --- internal state, lazily allocated on the first Step() call ---
	initialized bool

	grad, grad_old, x_old, q, s, y, searchDir *data.Slice
	s_point_vec, y_point_vec                  []*data.Slice
	rhoVals, alpha_LFBGS                      []float64

	f, f_old, H0k, gradNorm       float64
	eps, eps2, epsr, tolf2, tolf3 float64

	iter, globIter int
	converged      bool
	lastDm         fifoRing // reported to GUI as LastErr, same convention as Minimizer
}

// NewLBFGSMinimizer initializes a new L-BFGS minimizer configuration.
func NewLBFGSMinimizer(tolerance float64, maxIter int, verbose int) *LBFGSMinimizer {
	return &LBFGSMinimizer{
		Tolerance:    tolerance,
		MaxIter:      maxIter,
		Verbose:      verbose,
		History:      5,
		MaxStepAngle: LBFGSMaxStepAngle,
	}
}

// EnergyAndGradient updates the magnetization, normalizes it, and calculates
// both the system energy and the gradient.
func (l *LBFGSMinimizer) EnergyAndGradient(g *data.Slice) float64 {
	// 1. Ensure we are strictly on the unit sphere
	M.normalize()

	// 2. Compute the damping torque into g
	torqueFn(g)

	// 3. FIX THE SIGN!
	// LLNoPrecess calculates the damping torque as `-m x (m x B)`.
	// This evaluates to the NEGATIVE gradient (the descent direction).
	// L-BFGS requires the POSITIVE gradient to build its Hessian properly.
	cuda.Madd2(g, g, g, -1.0, 0.0)

	// 4. Return the total system energy
	return GetTotalEnergy()
}

// init lazily allocates persistent GPU buffers and history slots, and
// evaluates the initial energy/gradient. Mirrors the mini.k == nil pattern
// in Minimizer.Step().
func (l *LBFGSMinimizer) init() {
	m := M.Buffer()
	size := m.Size()

	l.grad = cuda.Buffer(3, size)
	l.grad_old = cuda.Buffer(3, size)
	l.x_old = cuda.Buffer(3, size)
	l.q = cuda.Buffer(3, size)
	l.s = cuda.Buffer(3, size)
	l.y = cuda.Buffer(3, size)
	l.searchDir = cuda.Buffer(3, size)

	mHist := l.History
	l.s_point_vec = make([]*data.Slice, mHist)
	l.y_point_vec = make([]*data.Slice, mHist)
	for i := 0; i < mHist; i++ {
		l.s_point_vec[i] = cuda.Buffer(3, size)
		l.y_point_vec[i] = cuda.Buffer(3, size)
	}
	l.rhoVals = make([]float64, mHist)
	l.alpha_LFBGS = make([]float64, mHist)

	l.eps = 1.1920929e-07 // float32 machine epsilon; energy/grad come from float32 GPU reductions
	l.eps2 = math.Sqrt(l.eps)
	l.epsr = math.Pow(l.eps, 0.9)
	l.tolf2 = math.Sqrt(l.Tolerance)
	l.tolf3 = math.Pow(l.Tolerance, 1.0/3.0)

	l.f = l.EnergyAndGradient(l.grad)
	l.gradNorm = float64(cuda.MaxVecNorm(l.grad))
	setMaxTorque(l.grad)

	l.H0k = 1.0
	l.iter = 0
	l.globIter = 0
	l.converged = false
	l.lastDm = FifoRing(DmSamples)

	if l.Verbose > 0 {
		fmt.Printf("objective function (energy)= %.24e\n", l.f)
	}

	if l.gradNorm < (l.epsr * (1.0 + math.Abs(l.f))) {
		l.converged = true
	}

	l.initialized = true
}

// Step executes a single outer L-BFGS iteration: two-loop recursion, line
// search, and history update. Satisfies the Stepper interface, so it is
// driven by RunWhile exactly like the steepest-descent Minimizer.
func (l *LBFGSMinimizer) Step() {
	if !l.initialized {
		l.init()
		if l.converged {
			return
		}
	}

	mHist := l.History
	l.f_old = l.f
	data.Copy(l.x_old, M.Buffer())
	data.Copy(l.grad_old, l.grad)
	data.Copy(l.q, l.grad)

	k := l.iter
	if mHist < k {
		k = mHist
	}

	// Backward pass
	for i := k - 1; i >= 0; i-- {
		l.rhoVals[i] = 1.0 / float64(cuda.Dot(l.s_point_vec[i], l.y_point_vec[i]))
		l.alpha_LFBGS[i] = l.rhoVals[i] * float64(cuda.Dot(l.s_point_vec[i], l.q))
		cuda.Madd2(l.q, l.q, l.y_point_vec[i], 1.0, float32(-l.alpha_LFBGS[i]))
	}

	// Scale q
	cuda.Madd2(l.q, l.q, l.q, float32(l.H0k), 0.0)

	// Forward pass
	for i := 0; i < k; i++ {
		beta := l.rhoVals[i] * float64(cuda.Dot(l.y_point_vec[i], l.q))
		cuda.Madd2(l.q, l.q, l.s_point_vec[i], 1.0, float32(l.alpha_LFBGS[i]-beta))
	}

	phiPrime0 := -float64(cuda.Dot(l.grad, l.q))
	isFirstIter := (l.globIter == 0)
	if phiPrime0 > 0 {
		data.Copy(l.q, l.grad)
		l.iter = 0
		isFirstIter = true // no curvature info survives this reset either
		phiPrime0 = -float64(cuda.Dot(l.grad, l.q))
		if l.Verbose > 2 {
			fmt.Println("descent ")
		}
	}

	cuda.Madd2(l.searchDir, l.q, l.q, -1.0, 0.0) // searchDir = -q

	var rate float64
	rate, l.f = l.linesearch(l.x_old, l.f, l.grad, l.searchDir, isFirstIter)
	if rate == 0.0 && l.Verbose > 0 {
		fmt.Println("Warning: LBFGS_Minimizer: linesearch returned rate == 0.0. This should not happen.")
	}

	f1 := 1.0 + math.Abs(l.f)
	l.gradNorm = float64(cuda.MaxVecNorm(l.grad))
	setMaxTorque(l.grad) // report to GUI

	if l.gradNorm < (l.epsr * f1) {
		l.converged = true
		return
	}

	// s = M.Buffer() - x_old
	cuda.Madd2(l.s, M.Buffer(), l.x_old, 1.0, -1.0)

	maxAbsSf32 := cuda.MaxVecNorm(l.s)
	maxAbsS := float64(maxAbsSf32)
	l.lastDm.Add(maxAbsSf32)
	setLastErr(l.lastDm.Max()) // report to GUI, same convention as Minimizer

	if l.Verbose > 1 {
		fmt.Printf("lbfgs> %d %.24e %e %e\n", l.globIter, l.f, l.gradNorm, rate)
	}

	maxAbsM := float64(cuda.MaxVecNorm(M.Buffer()))
	//check convergence criteria: relative change in energy, relative change in magnetization, and gradient norm
	if ((l.f_old - l.f) < (l.Tolerance * f1)) && (maxAbsS < (l.tolf2 * (1.0 + maxAbsM))) && (l.gradNorm <= (l.tolf3 * f1)) {
		l.converged = true
	}

	// y = grad - grad_old
	cuda.Madd2(l.y, l.grad, l.grad_old, 1.0, -1.0)
	ys := float64(cuda.Dot(l.y, l.s))

	normY := math.Sqrt(float64(cuda.Dot(l.y, l.y)))
	normS := math.Sqrt(float64(cuda.Dot(l.s, l.s)))

	//check Wolfe condition: s^T y > 0
	if ys <= l.eps2*normY*normS {
		if l.Verbose > 2 {
			fmt.Printf("%d WARNING: LBFGS_Minimizer:: skipping update!\n", l.iter)
		}
	} else {
		if l.iter < mHist {
			data.Copy(l.s_point_vec[l.iter], l.s)
			data.Copy(l.y_point_vec[l.iter], l.y)
		} else {
			sTmp := l.s_point_vec[0]
			yTmp := l.y_point_vec[0]
			for i := 0; i < mHist-1; i++ {
				l.s_point_vec[i] = l.s_point_vec[i+1]
				l.y_point_vec[i] = l.y_point_vec[i+1]
			}
			l.s_point_vec[mHist-1] = sTmp
			l.y_point_vec[mHist-1] = yTmp
			data.Copy(l.s_point_vec[mHist-1], l.s)
			data.Copy(l.y_point_vec[mHist-1], l.y)
		}
		l.H0k = ys / float64(cuda.Dot(l.y, l.y))
		l.iter++
	}

	l.globIter++
	NSteps++
}

// Free releases persistent GPU buffers. Satisfies the Stepper interface.
func (l *LBFGSMinimizer) Free() {
	if !l.initialized {
		return
	}
	l.grad.Free()
	l.grad_old.Free()
	l.x_old.Free()
	l.q.Free()
	l.s.Free()
	l.y.Free()
	l.searchDir.Free()
	for i := range l.s_point_vec {
		l.s_point_vec[i].Free()
		l.y_point_vec[i].Free()
	}
	l.initialized = false
}

// MinimizeLBFGS sets up the global environment, installs itself as the
// active Stepper, and drives itself via RunWhile -- identical wiring to
// Minimize(), so GUI refresh, table output, and pause/inject handling all
// come for free from runWhile.
func (l *LBFGSMinimizer) MinimizeLBFGS() bool {
	MinimizeConverged = false
	TimerStart := time.Now()

	// if wall-clock time is zero, skip minimization entirely (zero steps), and don't change any settings
	if MinimizeWallClockTime == 0 {
		return MinimizeConverged
	}

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

	Precess = false // disable precession for pure gradient calculation
	if stepper != nil {
		stepper.Free()
	}

	// install self as the active stepper, exactly like Minimize() does with mini
	stepper = l

	cond := func() bool {
		if !WallclockTimer(TimerStart, MinimizeWallClockTime) {
			return false
		}
		if l.globIter >= l.MaxIter {
			return false
		}
		if l.converged {
			return false
		}
		return true
	}

	RunWhile(cond)
	pause = true

	MinimizeConverged = l.converged
	stepper.Free()

	if l.globIter >= l.MaxIter && !l.converged && l.Verbose > 0 {
		fmt.Println("WARNING : maximum number of iterations exceeded in LBFGS")
	}

	return MinimizeConverged
}

func (l *LBFGSMinimizer) linesearch(x_old *data.Slice, fval float64, g *data.Slice, searchDir *data.Slice, isFirstIter bool) (rate, newF float64) {
	rate = 1.0
	if isFirstIter {
		gInfNorm := float64(cuda.MaxVecNorm(g))
		if gInfNorm > 1e-30 {
			rate = 1.0 / gInfNorm
		}
		if rate > 1.0 {
			rate = 1.0 // never take a larger-than-unit step with no curvature info yet
		}
	}
	newF, rate, _ = l.cvsrch(x_old, fval, g, rate, searchDir)
	return rate, newF
}

// lsPoint is one point evaluated during the line search: the step length,
// the objective value there, and the directional derivative there.
type lsPoint struct {
	stp, f, d float64
}

// cubicCoeffs computes the theta/gamma quantities used by MINPACK's
// safeguarded cubic interpolation formula. The algebra is identical in
// every case cstep uses it for; only the sign of gamma and the (p,q)
// pivot point differ from case to case.
func cubicCoeffs(fa, da, sa, fb, db, sb float64) (theta, gamma float64) {
	theta = 3.0*(fa-fb)/(sb-sa) + da + db
	s := math.Max(theta, math.Max(da, db))
	gamma = s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(da/s)*(db/s)))
	return theta, gamma
}

// cstep is MINPACK's safeguarded line-search step ("mcstep"): given the two
// current bracket endpoints x, y and a newly evaluated trial point t, it
// updates the bracket and proposes the next trial step.
//
// x and y hold the best and second-best points seen so far; t is the point
// just evaluated. Returns the updated x, y, the new trial step, whether the
// interval is now bracketed, and infoc, MINPACK's case code (1-4 normally;
// left at 0 if the inputs were inconsistent, matching the original's
// behavior of leaving *info untouched on early exit).
func cstep(x, y, t lsPoint, brackt bool, stpmin, stpmax float64) (newX, newY lsPoint, newStp float64, newBrackt bool, infoc int) {
	if (brackt && ((t.stp <= math.Min(x.stp, y.stp)) || (t.stp >= math.Max(x.stp, y.stp)))) || (x.d*(t.stp-x.stp) >= 0.0) || (stpmax < stpmin) {
		return x, y, t.stp, brackt, 0
	}

	sgnd := t.d * (x.d / math.Abs(x.d))
	bound := false
	var stpf, stpc, stpq float64

	switch {
	case t.f > x.f:
		infoc = 1
		bound = true
		theta, gamma := cubicCoeffs(x.f, x.d, x.stp, t.f, t.d, t.stp)
		if t.stp < x.stp {
			gamma = -gamma
		}
		p := (gamma - x.d) + theta
		q := ((gamma - x.d) + gamma) + t.d
		r := p / q
		stpc = x.stp + r*(t.stp-x.stp)
		stpq = x.stp + ((x.d/((x.f-t.f)/(t.stp-x.stp)+x.d))/2.0)*(t.stp-x.stp)
		if math.Abs(stpc-x.stp) < math.Abs(stpq-x.stp) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/2.0
		}
		brackt = true

	case sgnd < 0.0:
		infoc = 2
		theta, gamma := cubicCoeffs(x.f, x.d, x.stp, t.f, t.d, t.stp)
		if t.stp > x.stp {
			gamma = -gamma
		}
		p := (gamma - t.d) + theta
		q := ((gamma - t.d) + gamma) + x.d
		r := p / q
		stpc = t.stp + r*(x.stp-t.stp)
		stpq = t.stp + (t.d/(t.d-x.d))*(x.stp-t.stp)
		if math.Abs(stpc-t.stp) > math.Abs(stpq-t.stp) {
			stpf = stpc
		} else {
			stpf = stpq
		}
		brackt = true

	case math.Abs(t.d) < math.Abs(x.d):
		infoc = 3
		bound = true
		theta, gamma := cubicCoeffs(x.f, x.d, x.stp, t.f, t.d, t.stp)
		if t.stp > x.stp {
			gamma = -gamma
		}
		p := (gamma - t.d) + theta
		q := (gamma + (x.d - t.d)) + gamma
		r := p / q
		if (r < 0.0) && (gamma != 0.0) {
			stpc = t.stp + r*(x.stp-t.stp)
		} else if t.stp > x.stp {
			stpc = stpmax
		} else {
			stpc = stpmin
		}
		stpq = t.stp + (t.d/(t.d-x.d))*(x.stp-t.stp)
		if brackt {
			if math.Abs(t.stp-stpc) < math.Abs(t.stp-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		} else {
			if math.Abs(t.stp-stpc) > math.Abs(t.stp-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		}

	default:
		infoc = 4
		if brackt {
			theta, gamma := cubicCoeffs(t.f, t.d, t.stp, y.f, y.d, y.stp)
			if t.stp > y.stp {
				gamma = -gamma
			}
			p := (gamma - t.d) + theta
			q := ((gamma - t.d) + gamma) + y.d
			r := p / q
			stpc = t.stp + r*(y.stp-t.stp)
			stpf = stpc
		} else if t.stp > x.stp {
			stpf = stpmax
		} else {
			stpf = stpmin
		}
	}

	// Update the bracket endpoints with the newly evaluated trial point.
	if t.f > x.f {
		y = t
	} else {
		if sgnd < 0.0 {
			y = x
		}
		x = t
	}

	stpf = math.Min(stpmax, stpf)
	stpf = math.Max(stpmin, stpf)
	newStp = stpf

	if brackt && bound {
		if y.stp > x.stp {
			newStp = math.Min(x.stp+0.66*(y.stp-x.stp), newStp)
		} else {
			newStp = math.Max(x.stp+0.66*(y.stp-x.stp), newStp)
		}
	}

	return x, y, newStp, brackt, infoc
}

// cvsrch is MINPACK's More-Thuente line search, searching along searchDir
// from wa for a step satisfying the strong Wolfe conditions. wa is the base
// point, s the (fixed) search direction; f0/stp0 are the objective value
// and initial trial step. Returns the objective value and step length at
// the accepted point, and an info code (see the checks below; -1 marks
// early termination on bad input, matching the caller's existing handling).
func (l *LBFGSMinimizer) cvsrch(wa *data.Slice, f0 float64, g *data.Slice, stp0 float64, s *data.Slice) (newF, newStp float64, info int) {
	infoc := 1

	xtol := 1e-15
	ftol := 1.0e-4
	gtol := 0.9
	eps := 1.1920929e-07 // float32 machine epsilon (energy/gradient come from float32 GPU reductions)
	stpmin := 1e-15
	stpmax := 1e15
	xtrapf := 4.0
	maxfev := 20
	nfev := 0

	dginit := float64(cuda.Dot(g, s))
	if dginit >= 0.0 {
		if l.Verbose > 0 {
			fmt.Printf("WARNING: LBFGS_Minimizer:: no descent %e\n", dginit)
		}
		return f0, stp0, -1
	}

	// Precompute the angle-based step cap once, before trying any trial steps.
	// s (the search direction) is fixed for this whole call, only stp varies.
	stpAngleCap := math.Inf(1)
	if l.MaxStepAngle > 0 {
		maxDirNorm := float64(cuda.MaxVecNorm(s))
		if maxDirNorm > 0 {
			maxAngleRad := l.MaxStepAngle * math.Pi / 180.0
			stpAngleCap = math.Tan(maxAngleRad) / maxDirNorm
		}
	}

	brackt := false
	stage1 := true

	f := f0
	stp := stp0
	finit := f
	dgtest := ftol * dginit
	width := stpmax - stpmin
	width1 := 2.0 * width

	x := lsPoint{stp: 0.0, f: finit, d: dginit}
	y := lsPoint{stp: 0.0, f: finit, d: dginit}

	var stmin, stmax float64

	for {
		if brackt {
			stmin = math.Min(x.stp, y.stp)
			stmax = math.Max(x.stp, y.stp)
		} else {
			stmin = x.stp
			stmax = stp + xtrapf*(stp-x.stp)
		}

		stp = math.Max(stp, stpmin)
		stp = math.Min(stp, stpmax)

		if stp > stpAngleCap {
			stp = stpAngleCap
			if l.Verbose > 1 {
				fmt.Printf("LBFGS: step clamped by LBFGSMaxStepAngle (%.2f deg)\n", l.MaxStepAngle)
			}
		}

		if math.IsNaN(stp) || math.IsInf(stp, 0) {
			if l.Verbose > 0 {
				fmt.Println("WARNING: LBFGS_Minimizer: NaN/Inf step detected, resetting to stpmin")
			}
			stp = stpmin
		}

		if (brackt && ((stp <= stmin) || (stp >= stmax))) || (nfev >= maxfev-1) || (infoc == 0) || (brackt && (stmax-stmin <= xtol*stmax)) {
			stp = x.stp
		}

		// Update global M: M = wa + stp * s
		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(stp))

		// Evaluate updated objective
		f = l.EnergyAndGradient(g)
		nfev++

		dg := float64(cuda.Dot(g, s))
		ftest1 := finit + stp*dgtest
		ftest2 := finit + eps*math.Abs(finit)
		ft := 2.0*ftol - 1.0

		info = 0
		if (brackt && ((stp <= stmin) || (stp >= stmax))) || (infoc == 0) {
			info = 6
		}
		if (stp == stpmax) && (f <= ftest2) && (dg <= dgtest) {
			info = 5
		}
		if (stp == stpmin) && ((f > ftest2) || (dg >= dgtest)) {
			info = 4
		}
		if nfev >= maxfev {
			info = 3
		}
		if brackt && (stmax-stmin <= xtol*stmax) {
			info = 2
		}
		if (f <= ftest1) && (math.Abs(dg) <= gtol*(-dginit)) {
			info = 1
		}
		if (f <= ftest2) && (ft*dginit >= dg) && (math.Abs(dg) <= gtol*(-dginit)) {
			info = 1
		}

		if info != 0 {
			return f, stp, -1
		}

		if stage1 && (f <= ftest2) && (ft*dginit >= dg) && (dg >= math.Min(ftol, gtol)*dginit) {
			stage1 = false
		}

		t := lsPoint{stp: stp, f: f, d: dg}

		if stage1 && (f <= x.f) && !((f <= ftest2) && (ft*dginit >= dg)) {
			// Modified-function trick (subtract off the dgtest*step linear
			// term) while we haven't yet reached a point with a low enough
			// objective value -- same as MINPACK's cvsrch.
			xm := lsPoint{stp: x.stp, f: x.f - x.stp*dgtest, d: x.d - dgtest}
			ym := lsPoint{stp: y.stp, f: y.f - y.stp*dgtest, d: y.d - dgtest}
			tm := lsPoint{stp: t.stp, f: t.f - t.stp*dgtest, d: t.d - dgtest}

			var newXm, newYm lsPoint
			newXm, newYm, stp, brackt, infoc = cstep(xm, ym, tm, brackt, stmin, stmax)

			x = lsPoint{stp: newXm.stp, f: newXm.f + newXm.stp*dgtest, d: newXm.d + dgtest}
			y = lsPoint{stp: newYm.stp, f: newYm.f + newYm.stp*dgtest, d: newYm.d + dgtest}
		} else {
			x, y, stp, brackt, infoc = cstep(x, y, t, brackt, stmin, stmax)
		}

		if brackt {
			if math.Abs(y.stp-x.stp) >= 0.66*width1 {
				stp = x.stp + 0.5*(y.stp-x.stp)
			}
			width1 = width
			width = math.Abs(y.stp - x.stp)
		}
	}
}
