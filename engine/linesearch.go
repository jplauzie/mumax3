package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// This file holds the line-search machinery (strong-Wolfe via cvsrch/cstep,
// and Armijo backtracking) shared between LBFGSMinimizer and the
// steepest-descent Minimizer's inexact-line-search steps. Neither cvsrch
// nor armijoSearch depend on any minimizer-specific state -- they take an
// evalEG closure that updates M in place and returns the energy there,
// writing the gradient/torque into g as a side effect.
//
// evalEG's contract: given a *data.Slice g, evaluate energy and gradient at
// the CURRENT M.Buffer() (the caller is responsible for having moved M to
// the trial point first), write the gradient into g, and return the energy.

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
// (s) from wa for a step satisfying the strong Wolfe conditions. wa is the
// base point, s the (fixed) search direction; f0/stp0 are the objective
// value and initial trial step at wa. g must already hold the gradient at
// wa on entry (used for dginit) and will be overwritten with the gradient
// at the accepted point on return. evalEG moves M to the trial point
// (M = wa + stp*s), evaluates energy/gradient there, and returns the
// energy. Returns the objective value and step length at the accepted
// point, and an info code (-1 marks early termination on bad input).
func cvsrch(wa *data.Slice, f0 float64, g *data.Slice, stp0 float64, s *data.Slice,
	evalEG func(*data.Slice) float64, verbose int, maxStepAngle float64) (newF, newStp float64, info int) {
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
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch (Wolfe):: no descent %e\n", dginit)
		}
		return f0, stp0, -1
	}

	// Precompute the angle-based step cap once, before trying any trial steps.
	// s (the search direction) is fixed for this whole call, only stp varies.
	stpAngleCap := math.Inf(1)
	if maxStepAngle > 0 {
		maxDirNorm := float64(cuda.MaxVecNorm(s))
		if maxDirNorm > 0 {
			maxAngleRad := maxStepAngle * math.Pi / 180.0
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
			if verbose > 1 {
				fmt.Printf("linesearch: step clamped by MaxStepAngle (%.2f deg)\n", maxStepAngle)
			}
		}

		if math.IsNaN(stp) || math.IsInf(stp, 0) {
			if verbose > 0 {
				fmt.Println("WARNING: linesearch: NaN/Inf step detected, resetting to stpmin")
			}
			stp = stpmin
		}

		if (brackt && ((stp <= stmin) || (stp >= stmax))) || (nfev >= maxfev-1) || (infoc == 0) || (brackt && (stmax-stmin <= xtol*stmax)) {
			stp = x.stp
		}

		// Update global M: M = wa + stp * s
		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(stp))

		// Evaluate updated objective
		f = evalEG(g)
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

// armijoSearch performs backtracking line search along s from wa, accepting
// the first trial step satisfying the Armijo sufficient-decrease condition
// (with a relative-noise-floor fallback for problems where dginit is large
// relative to the total energy scale -- see ftest2). Unlike cvsrch, it
// checks no curvature condition, so rejected trials only need the energy;
// evalEnergyOnly should be a cheap energy-only evaluator (no gradient/
// torque computation). g is left unchanged until the very end, where
// evalEG is called exactly once at the accepted step to bring the
// gradient up to date.
func armijoSearch(wa *data.Slice, f0 float64, g *data.Slice, stp0 float64, s *data.Slice,
	evalEnergyOnly func() float64, evalEG func(*data.Slice) float64,
	verbose int, maxStepAngle float64) (newF, newStp float64, info int) {
	ftol := 1.0e-4
	backtrackFactor := 0.5
	stpmin := 1e-15
	maxfev := 20
	eps := 1.1920929e-07 // float32 machine epsilon, same as cvsrch

	dginit := float64(cuda.Dot(g, s))
	if dginit >= 0.0 {
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch (Armijo):: no descent %e\n", dginit)
		}
		return f0, stp0, -1
	}

	stpAngleCap := math.Inf(1)
	if maxStepAngle > 0 {
		maxDirNorm := float64(cuda.MaxVecNorm(s))
		if maxDirNorm > 0 {
			maxAngleRad := maxStepAngle * math.Pi / 180.0
			stpAngleCap = math.Tan(maxAngleRad) / maxDirNorm
		}
	}

	finit := f0
	ftest2 := finit + eps*math.Abs(finit) // relative noise-floor fallback, same as cvsrch
	stp := stp0
	if stp > stpAngleCap {
		stp = stpAngleCap
		if verbose > 1 {
			fmt.Printf("linesearch (Armijo): step clamped by MaxStepAngle (%.2f deg)\n", maxStepAngle)
		}
	}

	var f float64
	nfev := 0
	for {
		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(stp))
		f = evalEnergyOnly()
		nfev++

		// Accept if either the classic absolute Armijo condition holds, or
		// energy is within float32 noise of not increasing at all -- the
		// latter matters because dginit (summed over the whole mesh) can be
		// orders of magnitude larger than the total energy itself, making
		// the strict absolute-decrease test practically unsatisfiable at
		// some problems' energy scale.
		if f <= finit+ftol*stp*dginit || f <= ftest2 {
			break
		}
		if nfev >= maxfev || stp <= stpmin {
			if verbose > 0 {
				fmt.Println("WARNING: linesearch (Armijo): backtracking exhausted, accepting last trial")
			}
			break
		}
		stp *= backtrackFactor
	}

	f = evalEG(g)
	return f, stp, 0
}
