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
	step, f, slope float64
}

// cubicCoeffs computes the theta/gamma quantities used by MINPACK's
// safeguarded cubic interpolation formula. The algebra is identical in
// every case cstep uses it for; only the sign of gamma and the (p,q)
// pivot point differ from case to case.
func cubicCoeffs(fa, da, sa, fb, db, sb float64) (theta, gamma float64) {
	theta = 3.0*(fa-fb)/(sb-sa) + da + db
	s := math.Max(math.Abs(theta), math.Max(math.Abs(da), math.Abs(db)))
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
// TODO(minpack2): this post-update 0.66 clamp is the MINPACK-1 (mcstep) form.
func cstep(bestpoint, otherendpoint, trialpoint lsPoint, bracketed bool, stpmin, stpmax float64) (newBestPoint, newOtherEndPoint lsPoint, newStep float64, newIsBracketed bool, casecode int) {
	//check:1 trial is inside bracket,2:slope points towards trial,3: maxStep>minStep, else return early
	// reconsider casecode0, MP2 doesn't have
	if (bracketed && ((trialpoint.step <= math.Min(bestpoint.step, otherendpoint.step)) || (trialpoint.step >= math.Max(bestpoint.step, otherendpoint.step)))) || (bestpoint.slope*(trialpoint.step-bestpoint.step) >= 0.0) || (stpmax < stpmin) {
		return bestpoint, otherendpoint, trialpoint.step, bracketed, 0
	}

	sgnd := trialpoint.slope * (bestpoint.slope / math.Abs(bestpoint.slope))
	bound := false
	var chosenstep, cubicstep, quadstep float64

	switch {
	//trial overshot, so minimum is bracketed between best and trial
	case trialpoint.f > bestpoint.f:
		casecode = 1
		bound = true
		theta, gamma := cubicCoeffs(bestpoint.f, bestpoint.slope, bestpoint.step, trialpoint.f, trialpoint.slope, trialpoint.step)
		if trialpoint.step < bestpoint.step {
			gamma = -gamma
		}
		p := (gamma - bestpoint.slope) + theta
		q := ((gamma - bestpoint.slope) + gamma) + trialpoint.slope
		r := p / q
		cubicstep = bestpoint.step + r*(trialpoint.step-bestpoint.step)
		quadstep = bestpoint.step + ((bestpoint.slope/((bestpoint.f-trialpoint.f)/(trialpoint.step-bestpoint.step)+bestpoint.slope))/2.0)*(trialpoint.step-bestpoint.step)
		if math.Abs(cubicstep-bestpoint.step) < math.Abs(quadstep-bestpoint.step) {
			chosenstep = cubicstep
		} else {
			chosenstep = cubicstep + (quadstep-cubicstep)/2.0
		}
		bracketed = true
	// slope is reversed, but still bracketed
	case sgnd < 0.0:
		casecode = 2
		theta, gamma := cubicCoeffs(bestpoint.f, bestpoint.slope, bestpoint.step, trialpoint.f, trialpoint.slope, trialpoint.step)
		if trialpoint.step > bestpoint.step {
			gamma = -gamma
		}
		p := (gamma - trialpoint.slope) + theta
		q := ((gamma - trialpoint.slope) + gamma) + bestpoint.slope
		r := p / q
		cubicstep = trialpoint.step + r*(bestpoint.step-trialpoint.step)
		quadstep = trialpoint.step + (trialpoint.slope/(trialpoint.slope-bestpoint.slope))*(bestpoint.step-trialpoint.step)
		if math.Abs(cubicstep-trialpoint.step) > math.Abs(quadstep-trialpoint.step) {
			chosenstep = cubicstep
		} else {
			chosenstep = quadstep
		}
		bracketed = true

		// slope smaller, bottoming out
	case math.Abs(trialpoint.slope) < math.Abs(bestpoint.slope):
		casecode = 3
		bound = true
		theta, gamma := cubicCoeffs(bestpoint.f, bestpoint.slope, bestpoint.step, trialpoint.f, trialpoint.slope, trialpoint.step)
		if trialpoint.step > bestpoint.step {
			gamma = -gamma
		}
		p := (gamma - trialpoint.slope) + theta
		q := (gamma + (bestpoint.slope - trialpoint.slope)) + gamma
		r := p / q
		if (r < 0.0) && (gamma != 0.0) {
			cubicstep = trialpoint.step + r*(bestpoint.step-trialpoint.step)
		} else if trialpoint.step > bestpoint.step {
			cubicstep = stpmax
		} else {
			cubicstep = stpmin
		}
		quadstep = trialpoint.step + (trialpoint.slope/(trialpoint.slope-bestpoint.slope))*(bestpoint.step-trialpoint.step)
		if bracketed {
			if math.Abs(trialpoint.step-cubicstep) < math.Abs(trialpoint.step-quadstep) {
				chosenstep = cubicstep
			} else {
				chosenstep = quadstep
			}
		} else {
			if math.Abs(trialpoint.step-cubicstep) > math.Abs(trialpoint.step-quadstep) {
				chosenstep = cubicstep
			} else {
				chosenstep = quadstep
			}
		}

	default:
		// still going downhill
		casecode = 4
		if bracketed {
			theta, gamma := cubicCoeffs(trialpoint.f, trialpoint.slope, trialpoint.step, otherendpoint.f, otherendpoint.slope, otherendpoint.step)
			if trialpoint.step > otherendpoint.step {
				gamma = -gamma
			}
			p := (gamma - trialpoint.slope) + theta
			q := ((gamma - trialpoint.slope) + gamma) + otherendpoint.slope
			r := p / q
			cubicstep = trialpoint.step + r*(otherendpoint.step-trialpoint.step)
			chosenstep = cubicstep
		} else if trialpoint.step > bestpoint.step {
			chosenstep = stpmax
		} else {
			chosenstep = stpmin
		}
	}

	// Update the bracket endpoints with the newly evaluated trial point.
	if trialpoint.f > bestpoint.f {
		otherendpoint = trialpoint
	} else {
		if sgnd < 0.0 {
			otherendpoint = bestpoint
		}
		bestpoint = trialpoint
	}

	chosenstep = math.Min(stpmax, chosenstep)
	chosenstep = math.Max(stpmin, chosenstep)
	newStep = chosenstep

	if bracketed && bound {
		if otherendpoint.step > bestpoint.step {
			newStep = math.Min(bestpoint.step+0.66*(otherendpoint.step-bestpoint.step), newStep)
		} else {
			newStep = math.Max(bestpoint.step+0.66*(otherendpoint.step-bestpoint.step), newStep)
		}
	}

	return bestpoint, otherendpoint, newStep, bracketed, casecode
}

// MTlinesearch is MINPACK's More-Thuente line search, searching along searchDir
// (s) from wa for a step satisfying the strong Wolfe conditions. wa is the
// base point, s the (fixed) search direction; f0/stp0 are the objective
// value and initial trial step at wa. g must already hold the gradient at
// wa on entry (used for dginit) and will be overwritten with the gradient
// at the accepted point on return. evalEG moves M to the trial point
// (M = wa + stp*s), evaluates energy/gradient there, and returns the
// energy. Returns the objective value and step length at the accepted
// point, and an info code (-1 marks early termination on bad input).

// Line-search function and its derivative (More-Thuente notation):
//
//	phi(a)  = f(m0 + a*d)
//	phi'(a) = grad f(m0 + a*d) . d
//
// initialSlope = phi'(0), trialSlope = phi'(step), initialEnergy = phi(0).
func MTlinesearch(wa *data.Slice, f0 float64, g *data.Slice, step0 float64, s *data.Slice,
	evalEG func(*data.Slice) float64, verbose int, maxStepAngle float64) (newF, newStp float64, info int) {
	casecode := 1 //case code for cstep

	// Strong Wolfe conditions (mu = ftol, eta = gtol):
	//   sufficient decrease:  phi(a) <= phi(0) + mu * a * phi'(0)
	//   curvature:            |phi'(a)| <= eta * |phi'(0)|

	bracketWidthTol := 1e-7 //  relative tolerance on [bracketLow, bracketHigh]. Once the bracket is narrower than bracketWidthTol * bracketHigh, the search stops (info = 2).
	ftol := 1.0e-4          // (Wolfe c1 in N&W) sets how much energy drop a trial step must
	// deliver. A step is accepted only if the energy falls by at least ftol * step * initialSlope.
	gtol := 0.9 // tunable. gtol (Wolfe c2) sets how flat the energy must be at the accepted
	// point. The slope there must satisfy |slope| <= gtol * |initialSlope|,
	// which rules out steps that stop while the energy is still dropping steeply.
	// 0.9 is a loose test, the usual choice for quasi-Newton methods.
	f32eps := 1.1920929e-07 // float32 machine epsilon (energy/gradient come from float32 GPU reductions)
	minStep := 1e-15
	maxStep := 1e15
	extrapfactor := 4.0 //The next trial may be at most step + extrapFactor*(step - bestStep), i.e. about 5x the current step when starting from bestStep = 0.
	maxEvals := 20
	numEvals := 0

	slopeinit := float64(cuda.Dot(g, s)) // slopeInit (was dginit) = phi'(0) = g . s. Must be negative (s is a descent direction).
	if slopeinit >= 0.0 {
		//not a descent, return immediately
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch (Wolfe):: no descent %e\n", slopeinit)
		}
		return f0, step0, -1
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

	bracketed := false
	stage1 := true

	f := f0
	// alpha in MT(1994)
	step := step0
	f_init := f
	dgtest := ftol * slopeinit
	width := maxStep - minStep
	width1 := 2.0 * width

	bestPoint := lsPoint{step: 0.0, f: f_init, slope: slopeinit}
	otherEndPoint := lsPoint{step: 0.0, f: f_init, slope: slopeinit}

	var stmin, stmax float64

	for {
		if bracketed == true {
			stmin = math.Min(bestPoint.step, otherEndPoint.step)
			stmax = math.Max(bestPoint.step, otherEndPoint.step)
		} else {
			stmin = bestPoint.step
			stmax = step + extrapfactor*(step-bestPoint.step)
		}

		step = math.Max(step, minStep)
		step = math.Min(step, maxStep)

		if step > stpAngleCap {
			step = stpAngleCap
			if verbose > 1 {
				fmt.Printf("linesearch: step clamped by MaxStepAngle (%.2f deg)\n", maxStepAngle)
			}
		}

		if math.IsNaN(step) || math.IsInf(step, 0) {
			if verbose > 0 {
				fmt.Println("WARNING: linesearch: NaN/Inf step detected, resetting to minStep")
			}
			step = minStep
		}

		if (bracketed && ((step <= stmin) || (step >= stmax))) || (numEvals >= maxEvals-1) || (casecode == 0) || (bracketed && (stmax-stmin <= bracketWidthTol*stmax)) {
			step = bestPoint.step
		}

		// Update global M: M = wa + step * s
		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(step))

		// Evaluate updated objective
		// Normalize M, recalculate new torque (gradient, watching minus sign convention) and energy at the trial point.
		f = evalEG(g)
		numEvals++

		slope := float64(cuda.Dot(g, s))
		ftest1 := f_init + step*dgtest
		noisefloor := f_init + f32eps*math.Abs(f_init)
		//revisit this
		ft := 2.0*ftol - 1.0

		// 1:success, 2:bracket collapsed,3:numEvals>=maxEvals,4:stuck at minStep,5:stuck at maxStep, 6:step at or beyond end of bracket or cstep fail
		//	cstep can fail if:trial outside bracket, slope doesn't point towards trial, step limits inverted
		info = 0
		if (bracketed && ((step <= stmin) || (step >= stmax))) || (casecode == 0) {
			info = 6
		}
		if (step == maxStep) && (f <= noisefloor) && (slope <= dgtest) {
			info = 5
		}
		if (step == minStep) && ((f > noisefloor) || (slope >= dgtest)) {
			info = 4
		}
		if numEvals >= maxEvals {
			info = 3
		}
		if bracketed && (stmax-stmin <= bracketWidthTol*stmax) {
			info = 2
		}
		if (f <= ftest1) && (math.Abs(slope) <= gtol*(-slopeinit)) {
			info = 1
		}
		if (f <= noisefloor) && (ft*slopeinit >= slope) && (math.Abs(slope) <= gtol*(-slopeinit)) {
			info = 1
		}

		//change -1 return val
		if info != 0 {
			return f, step, -1
		}

		//we're done with stage1
		//recheck f<=noisefloor, seems wrong
		if stage1 && (f <= noisefloor) && (ft*slopeinit >= slope) && (slope >= math.Min(ftol, gtol)*slopeinit) {
			stage1 = false
		}

		trialpoint := lsPoint{step: step, f: f, slope: slope}

		if stage1 && (f <= bestPoint.f) && !((f <= noisefloor) && (ft*slopeinit >= slope)) {
			// Auxiliary function psi (stage 1 works on psi instead of phi):
			//   psi(a)  = phi(a) - phi(0) - mu * phi'(0) * a
			//   psi'(a) = phi'(a) - mu * phi'(0)
			// psi(a) <= 0 is exactly the sufficient decrease condition. In code the
			// constant phi(0) is dropped: energy - step*sufficientDecreaseSlope.
			// Modified-function trick ψ(subtract off the dgtest*step linear
			// term) while we haven't yet reached a point with a low enough
			// objective value -- same as MINPACK's cvsrch.
			bestpointmod := lsPoint{step: bestPoint.step, f: bestPoint.f - bestPoint.step*dgtest, slope: bestPoint.slope - dgtest}
			otherendpointmod := lsPoint{step: otherEndPoint.step, f: otherEndPoint.f - otherEndPoint.step*dgtest, slope: otherEndPoint.slope - dgtest}
			trialpointmod := lsPoint{step: trialpoint.step, f: trialpoint.f - trialpoint.step*dgtest, slope: trialpoint.slope - dgtest}

			var newbestpointmod, newotherendpointmod lsPoint
			newbestpointmod, newotherendpointmod, step, bracketed, casecode = cstep(bestpointmod, otherendpointmod, trialpointmod, bracketed, stmin, stmax)

			//restore φ
			bestPoint = lsPoint{step: newbestpointmod.step, f: newbestpointmod.f + newbestpointmod.step*dgtest, slope: newbestpointmod.slope + dgtest}
			otherEndPoint = lsPoint{step: newotherendpointmod.step, f: newotherendpointmod.f + newotherendpointmod.step*dgtest, slope: newotherendpointmod.slope + dgtest}
		} else {
			//φ
			bestPoint, otherEndPoint, step, bracketed, casecode = cstep(bestPoint, otherEndPoint, trialpoint, bracketed, stmin, stmax)
		}

		//fallback bisection after bracketing. if bracket doesn't shrink by at least 33% after 2 passes, throwaway cstep proposal and bisect between best and otherEndpoint
		if bracketed {
			if math.Abs(otherEndPoint.step-bestPoint.step) >= 0.66*width1 {
				step = bestPoint.step + 0.5*(otherEndPoint.step-bestPoint.step)
			}
			width1 = width
			width = math.Abs(otherEndPoint.step - bestPoint.step)
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
