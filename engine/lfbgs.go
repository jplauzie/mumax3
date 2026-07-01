package engine

import (
	"fmt"
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Based heavily on "Numerical Optimization" by Nocedal and Wright, 2nd edition, Springer, 2006.
//see also  J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage (1980), Mathematics of Computation 35, pp. 773-782.

var (
	LBFGSTolerance float64 = 1e-5
	LBFGSMaxIter   int     = 10000
	LBFGSVerbose   int     = 0
	//Nocedal suggests between 3-20
	LBFGSHistory      int     = 5
	LBFGSMaxStepAngle float64 = 45.0 // max degrees any cell's m may rotate in one trial step (<=0 disables)
)

// 2. Create a zero-argument wrapper function for the script to call
func RunMinimizeLBFGS() bool {
	// Instantiate the minimizer using the current global script variables
	minimizer := NewLBFGSMinimizer(LBFGSTolerance, LBFGSMaxIter, LBFGSVerbose)
	minimizer.History = LBFGSHistory
	minimizer.MaxStepAngle = LBFGSMaxStepAngle

	// Execute the minimization
	return minimizer.MinimizeLBFGS()
}

// 3. Register the function and variables with the MuMax3 script parser
func init() {
	// Expose the main function
	DeclFunc("MinimizeLBFGS", RunMinimizeLBFGS, "Use the L-BFGS method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded.")

	// Expose the configurable variables
	DeclVar("LBFGSTolerance", &LBFGSTolerance, "Convergence tolerance for the L-BFGS minimizer (default: 1e-5).")
	DeclVar("LBFGSMaxIter", &LBFGSMaxIter, "Maximum number of iterations for the L-BFGS minimizer (default: 10000).")
	DeclVar("LBFGSVerbose", &LBFGSVerbose, "Verbosity level of the L-BFGS minimizer: 0=silent, 1=basic, 2=detailed (default: 0).")
	DeclVar("LBFGSHistory", &LBFGSHistory, "Number of previous gradients to store for the L-BFGS inverse Hessian approximation (default: 5).")
	DeclVar("LBFGSMaxStepAngle", &LBFGSMaxStepAngle, "Maximum angle (degrees) magnetization may rotate in a single L-BFGS trial step; prevents jumping to an unrelated energy basin. Set <=0 to disable (default: 45).")
}

// LBFGSMinimizer implements the L-BFGS optimization routine.
type LBFGSMinimizer struct {
	Tolerance    float64
	MaxIter      int
	Verbose      int
	History      int // 'm' parameter in L-BFGS (default: 5)
	MaxStepAngle float64
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
// both the system energy and the tangent-space gradient.
// EnergyAndGradient updates the magnetization, normalizes it, and calculates
// both the system energy and the gradient.
func (l *LBFGSMinimizer) EnergyAndGradient(g *data.Slice) float64 {
	// Ensure we are strictly on the unit sphere
	M.normalize()

	// Compute the damping torque into g
	torqueFn(g)
	setMaxTorque(g)

	// Sign is backwards, as in minimizer
	// LLNoPrecess calculates the damping torque as `-m x (m x B)`.
	cuda.Madd2(g, g, g, -1.0, 0.0)

	return GetTotalEnergy()
}

// MinimizeLBFGS securely sets up the global environment and executes the L-BFGS loop.
func (l *LBFGSMinimizer) MinimizeLBFGS() bool {
	// Setup standard minimization environment conventions
	TimerStart := time.Now()
	MinimizeConverged = false

	// if wall-clock time is zero, skip minimization entirely (zero steps), and don't change any settings
	if MinimizeWallClockTime == 0 {
		MinimizeConverged = false
		return MinimizeConverged
	}

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
		stepper = nil
	}

	eps64 := 2.22e-16
	sqrteps := math.Sqrt(eps64)
	epsr := math.Pow(eps64, 0.9)
	tolf2 := math.Sqrt(l.Tolerance)
	tolf3 := math.Pow(l.Tolerance, 1.0/3.0)

	m := M.Buffer()
	size := m.Size()

	grad := cuda.Buffer(3, size)
	defer grad.Free()

	f := l.EnergyAndGradient(grad)
	gradNorm := float64(cuda.MaxVecNorm(grad))

	if l.Verbose > 0 {
		fmt.Printf("objective function (energy)= %.24e\n", f)
	}
	if gradNorm < (epsr * (1.0 + math.Abs(f))) {
		MinimizeConverged = true
		return MinimizeConverged
	}
	//minimize further with torque? or use dm criteria?

	mHist := l.History
	s_point_vec := make([]*data.Slice, mHist)
	y_point_vec := make([]*data.Slice, mHist)
	for i := 0; i < mHist; i++ {
		s_point_vec[i] = cuda.Buffer(3, size)
		y_point_vec[i] = cuda.Buffer(3, size)
		defer s_point_vec[i].Free()
		defer y_point_vec[i].Free()
	}

	alpha_LFBGS := make([]float64, mHist)

	q := cuda.Buffer(3, size)
	s := cuda.Buffer(3, size)
	y := cuda.Buffer(3, size)
	grad_old := cuda.Buffer(3, size)
	x_old := cuda.Buffer(3, size)
	searchDir := cuda.Buffer(3, size)
	rhoVals := make([]float64, mHist)

	defer q.Free()
	defer s.Free()
	defer y.Free()
	defer grad_old.Free()
	defer x_old.Free()
	defer searchDir.Free()

	f_old := 0.0
	iter := 0
	globIter := 0
	H0k := 1.0

	for globIter < l.MaxIter && WallclockTimer(TimerStart, MinimizeWallClockTime) {
		//cgSteps := 0
		f_old = f
		data.Copy(x_old, M.Buffer())
		data.Copy(grad_old, grad)
		data.Copy(q, grad)

		k := iter
		if mHist < k {
			k = mHist
		}

		// Backward Pass
		for i := k - 1; i >= 0; i-- {
			rhoVals[i] = 1.0 / float64(cuda.Dot(s_point_vec[i], y_point_vec[i]))
			alpha_LFBGS[i] = rhoVals[i] * float64(cuda.Dot(s_point_vec[i], q))
			// q = q - alpha[i]*yVector[i]
			cuda.Madd2(q, q, y_point_vec[i], 1.0, float32(-alpha_LFBGS[i]))
		}

		// Scale q
		cuda.Madd2(q, q, q, float32(H0k), 0.0)

		// Forward Pass
		for i := 0; i < k; i++ {
			//rho := 1.0 / float64(cuda.Dot(s_point_vec[i], y_point_vec[i]))
			beta := rhoVals[i] * float64(cuda.Dot(y_point_vec[i], q))
			// q = q + sVector[i]*(alpha[i] - beta)
			cuda.Madd2(q, q, s_point_vec[i], 1.0, float32(alpha_LFBGS[i]-beta))
		}

		phiPrime0 := -float64(cuda.Dot(grad, q))
		isFirstIter := (globIter == 0)
		if phiPrime0 > 0 {
			data.Copy(q, grad)
			iter = 0
			isFirstIter = true // no curvature info survives this reset either
			if l.Verbose > 2 {
				fmt.Println("descent ")
			}
		}

		//is float64 epsilon correct? or should it be float32?

		if -float64(cuda.Dot(grad, q)) > -1e-15 {
			gradNorm = float64(cuda.MaxVecNorm(grad))
			if gradNorm < eps64*(1.0+math.Abs(f)) && l.Verbose > 0 {
				fmt.Printf("Minimizer: Convergence reached (due to almost zero gradient (|grad|=%e < %e)!\n", gradNorm, eps64*(1.0+math.Abs(f)))
				MinimizeConverged = false
				return MinimizeConverged
			}
		}

		cuda.Madd2(searchDir, q, q, -1.0, 0.0) // searchDir = -q

		rate := l.linesearch(x_old, &f, grad, searchDir, isFirstIter)

		f1 := 1.0 + math.Abs(f)
		gradNorm = float64(cuda.MaxVecNorm(grad))
		if gradNorm < (epsr * f1) {
			MinimizeConverged = true
			return MinimizeConverged
		}

		// s = M.Buffer() - x_old
		cuda.Madd2(s, M.Buffer(), x_old, 1.0, -1.0)

		if l.Verbose > 1 {
			fmt.Printf("lbfgs> %d %.24e %e %e\n", globIter, f, gradNorm, rate)
		}

		maxAbsS := float64(cuda.MaxVecNorm(s))
		maxAbsM := float64(cuda.MaxVecNorm(M.Buffer()))
		//check convergence criteria: relative change in energy, relative change in magnetization, and gradient norm
		//2nd condition probably should just be tolf2?
		//3rd condition should probably just be dM or maxTorque
		if ((f_old - f) < (l.Tolerance * f1)) && (maxAbsS < (tolf2 * (1.0 + maxAbsM))) && (gradNorm <= (tolf3 * f1)) {
			MinimizeConverged = true
			break
		}

		// y = grad - grad_old
		cuda.Madd2(y, grad, grad_old, 1.0, -1.0)
		ys := float64(cuda.Dot(y, s))

		normY := math.Sqrt(float64(cuda.Dot(y, y)))
		normS := math.Sqrt(float64(cuda.Dot(s, s)))

		//Nocedal suggests not skipping the update, because LFBGS can be selfcorrecting. test further (need an actual example where this fires)
		//check Wolfe condition: s^T y > 0
		if ys <= sqrteps*normY*normS {
			if l.Verbose > 2 {
				fmt.Printf("%d WARNING: LBFGS_Minimizer:: skipping update!\n", iter)
			}
		} else {
			if iter < mHist {
				// history vectors not full yet, just add new vectors
				data.Copy(s_point_vec[iter], s)
				data.Copy(y_point_vec[iter], y)
			} else {
				// history vectors full, shift all vectors in a FIFO queue
				sTmp := s_point_vec[0]
				yTmp := y_point_vec[0]
				for i := 0; i < mHist-1; i++ {
					s_point_vec[i] = s_point_vec[i+1]
					y_point_vec[i] = y_point_vec[i+1]
				}
				s_point_vec[mHist-1] = sTmp
				y_point_vec[mHist-1] = yTmp
				data.Copy(s_point_vec[mHist-1], s)
				data.Copy(y_point_vec[mHist-1], y)
			}
			H0k = ys / float64(cuda.Dot(y, y))
			iter++
		}

		globIter++
		NSteps++
	}

	if globIter >= l.MaxIter && l.Verbose > 0 {
		fmt.Println("WARNING : maximum number of iterations exceeded in LBFGS")
	}

	return MinimizeConverged
}

func (l *LBFGSMinimizer) linesearch(x_old *data.Slice, fval *float64, g *data.Slice, searchDir *data.Slice, isFirstIter bool) float64 {
	rate := 1.0
	if isFirstIter {
		gInfNorm := float64(cuda.MaxVecNorm(g))
		if gInfNorm > 0 {
			rate = 1.0 / gInfNorm
		}
		if rate > 1.0 {
			rate = 1.0 // never take a larger-than-unit step with no curvature info yet
		}
	}
	l.cubicvalsearch(x_old, fval, g, &rate, searchDir)
	if rate == 0.0 && l.Verbose > 0 {
		fmt.Println("Warning: LBFGS_Minimizer: linesearch returned rate == 0.0!")
	}
	return rate
}

func (l *LBFGSMinimizer) cubicvalsearch(wa *data.Slice, f *float64, g *data.Slice, stp *float64, s *data.Slice) int {
	info := 0
	infoc := 1

	xtol := 1e-15
	ftol := 1.0e-4
	gtol := 0.9
	eps32 := 1.1920929e-07 // float32 machine epsilon (energy/gradient come from float32 GPU reductions)
	stepmin := 1e-15
	stepmax := 1e15
	xtrapf := 4.0
	max_f_ev := 20
	num_f_evals := 0

	dginit := float64(cuda.Dot(g, s))
	if dginit >= 0.0 {
		if l.Verbose > 0 {
			fmt.Printf("WARNING: LBFGS_Minimizer:: no descent %e\n", dginit)
		}
		return -1
	}

	// Precompute the angle-based step cap once, before trying any trial steps.
	// For a unit vector m, adding a component of magnitude stp*||s||_inf and
	// renormalizing rotates it by approximately atan(stp*||s||_inf). This is a
	// conservative (upper-bound) estimate since it ignores the component of s
	// parallel to m. s (the search direction) is fixed for this whole call,
	// only *stp varies, so this only needs to be computed once.
	stpAngleCap := math.Inf(1)
	if l.MaxStepAngle > 0 {
		maxDirNorm := float64(cuda.MaxVecNorm(s))
		if maxDirNorm > 0 {
			maxAngleRad := l.MaxStepAngle * math.Pi / 180.0
			stpAngleCap = math.Tan(maxAngleRad) / maxDirNorm
		}
	}

	bracketed := false
	stage1 := true

	finit := *f
	dgtest := ftol * dginit
	width := stepmax - stepmin
	width1 := 2.0 * width

	stx := 0.0
	fx := finit
	dgx := dginit
	sty := 0.0
	fy := finit
	dgy := dginit

	stmin := math.NaN()
	stmax := math.NaN()

	//while loop in Go
	for {
		if bracketed {
			stmin = math.Min(stx, sty)
			stmax = math.Max(stx, sty)
		} else {
			stmin = stx
			stmax = *stp + xtrapf*(*stp-stx)
		}

		*stp = math.Max(*stp, stepmin)
		*stp = math.Min(*stp, stepmax)

		if *stp > stpAngleCap {
			*stp = stpAngleCap
			if l.Verbose > 1 {
				fmt.Printf("LBFGS: step clamped by LBFGSMaxStepAngle (%.2f deg)\n", l.MaxStepAngle)
			}
		}

		if (bracketed && ((*stp <= stmin) || (*stp >= stmax))) || (num_f_evals >= max_f_ev-1) || (infoc == 0) || (bracketed && (stmax-stmin <= xtol*stmax)) {
			*stp = stx
		}

		// Update global M: M = wa + stp * s
		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(*stp))

		// Evaluate updated objective
		*f = l.EnergyAndGradient(g)

		num_f_evals++

		dg := float64(cuda.Dot(g, s))
		ftest1 := finit + (*stp)*dgtest
		ftest2 := finit + eps32*math.Abs(finit)
		ft := 2.0*ftol - 1.0

		//

		if (bracketed && ((*stp <= stmin) || (*stp >= stmax))) || (infoc == 0) {
			info = 6
		}
		if (*stp == stepmax) && (*f <= ftest2) && (dg <= dgtest) {
			info = 5
		}
		if (*stp == stepmin) && ((*f > ftest2) || (dg >= dgtest)) {
			info = 4
		}
		if num_f_evals >= max_f_ev {
			info = 3
		}
		if bracketed && (stmax-stmin <= xtol*stmax) {
			info = 2
		}
		if (*f <= ftest1) && (math.Abs(dg) <= gtol*(-dginit)) {
			info = 1
		}
		if (*f <= ftest2) && (ft*dginit >= dg) && (math.Abs(dg) <= gtol*(-dginit)) {
			info = 1
		}

		if info != 0 {
			return -1
		}

		if stage1 && (*f <= ftest2) && (ft*dginit >= dg) && (dg >= math.Min(ftol, gtol)*dginit) {
			stage1 = false
		}

		if stage1 && (*f <= fx) && !((*f <= ftest2) && (ft*dginit >= dg)) {
			fm := *f - (*stp)*dgtest
			fxm := fx - stx*dgtest
			fym := fy - sty*dgtest
			dgm := dg - dgtest
			dgxm := dgx - dgtest
			dgym := dgy - dgtest

			cubicstep(&stx, &fxm, &dgxm, &sty, &fym, &dgym, stp, &fm, &dgm, &bracketed, stmin, stmax, &infoc)

			fx = fxm + stx*dgtest
			fy = fym + sty*dgtest
			dgx = dgxm + dgtest
			dgy = dgym + dgtest
		} else {
			cubicstep(&stx, &fx, &dgx, &sty, &fy, &dgy, stp, f, &dg, &bracketed, stmin, stmax, &infoc)
		}

		if bracketed {
			if math.Abs(sty-stx) >= 0.66*width1 {
				*stp = stx + 0.5*(sty-stx)
			}
			width1 = width
			width = math.Abs(sty - stx)
		}
	}
}

func cubicstep(stx, fx, dx, sty, fy, dy, stp, fp, dp *float64, bracketed *bool, stpmin, stpmax float64, info *int) int {
	*info = 0
	bound := false

	if (*bracketed && ((*stp <= math.Min(*stx, *sty)) || (*stp >= math.Max(*stx, *sty)))) || (*dx*(*stp-*stx) >= 0.0) || (stpmax < stpmin) {
		return -1
	}

	sgnd := (*dp) * (*dx / math.Abs(*dx))

	var stpf, stpc, stpq float64

	if *fp > *fx {
		*info = 1
		bound = true
		theta := 3.0*(*fx-*fp)/(*stp-*stx) + *dx + *dp
		s := math.Max(theta, math.Max(*dx, *dp))
		gamma := s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(*dx/s)*(*dp/s)))
		if *stp < *stx {
			gamma = -gamma
		}
		p := (gamma - *dx) + theta
		q := ((gamma - *dx) + gamma) + *dp
		r := p / q
		stpc = *stx + r*(*stp-*stx)
		stpq = *stx + ((*dx/((*fx-*fp)/(*stp-*stx)+*dx))/2.0)*(*stp-*stx)
		if math.Abs(stpc-*stx) < math.Abs(stpq-*stx) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/2.0
		}
		*bracketed = true
	} else if sgnd < 0.0 {
		*info = 2
		bound = false
		theta := 3.0*(*fx-*fp)/(*stp-*stx) + *dx + *dp
		s := math.Max(theta, math.Max(*dx, *dp))
		gamma := s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(*dx/s)*(*dp/s)))
		if *stp > *stx {
			gamma = -gamma
		}
		p := (gamma - *dp) + theta
		q := ((gamma - *dp) + gamma) + *dx
		r := p / q
		stpc = *stp + r*(*stx-*stp)
		stpq = *stp + (*dp/(*dp-*dx))*(*stx-*stp)
		if math.Abs(stpc-*stp) > math.Abs(stpq-*stp) {
			stpf = stpc
		} else {
			stpf = stpq
		}
		*bracketed = true
	} else if math.Abs(*dp) < math.Abs(*dx) {
		*info = 3
		bound = true
		theta := 3.0*(*fx-*fp)/(*stp-*stx) + *dx + *dp
		s := math.Max(theta, math.Max(*dx, *dp))
		gamma := s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(*dx/s)*(*dp/s)))
		if *stp > *stx {
			gamma = -gamma
		}
		p := (gamma - *dp) + theta
		q := (gamma + (*dx - *dp)) + gamma
		r := p / q
		if (r < 0.0) && (gamma != 0.0) {
			stpc = *stp + r*(*stx-*stp)
		} else if *stp > *stx {
			stpc = stpmax
		} else {
			stpc = stpmin
		}
		stpq = *stp + (*dp/(*dp-*dx))*(*stx-*stp)
		if *bracketed {
			if math.Abs(*stp-stpc) < math.Abs(*stp-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		} else {
			if math.Abs(*stp-stpc) > math.Abs(*stp-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		}
	} else {
		*info = 4
		bound = false
		if *bracketed {
			theta := 3.0*(*fp-*fy)/(*sty-*stp) + *dy + *dp
			s := math.Max(theta, math.Max(*dy, *dp))
			gamma := s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(*dy/s)*(*dp/s)))
			if *stp > *sty {
				gamma = -gamma
			}
			p := (gamma - *dp) + theta
			q := ((gamma - *dp) + gamma) + *dy
			r := p / q
			stpc = *stp + r*(*sty-*stp)
			stpf = stpc
		} else if *stp > *stx {
			stpf = stpmax
		} else {
			stpf = stpmin
		}
	}

	if *fp > *fx {
		*sty = *stp
		*fy = *fp
		*dy = *dp
	} else {
		if sgnd < 0.0 {
			*sty = *stx
			*fy = *fx
			*dy = *dx
		}
		*stx = *stp
		*fx = *fp
		*dx = *dp
	}

	stpf = math.Min(stpmax, stpf)
	stpf = math.Max(stpmin, stpf)
	*stp = stpf

	if *bracketed && bound {
		if *sty > *stx {
			*stp = math.Min(*stx+0.66*(*sty-*stx), *stp)
		} else {
			*stp = math.Max(*stx+0.66*(*sty-*stx), *stp)
		}
	}

	return 0
}
