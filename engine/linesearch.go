package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// This file holds the line-search machinery (strong-Wolfe) shared between LBFGSMinimizer and the
// steepest-descent Minimizer's inexact-line-search steps. MTlinsearch does not depend on any minimizer-specific state -- they take an
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
func cstep(bestpoint, otherendpoint, trialpoint lsPoint, bracketed bool, stpmin, stpmax, fnoise float64) (newBestPoint, newOtherEndPoint lsPoint, newStep float64, newIsBracketed bool, casecode int) {
	//check:1 trial is inside bracket,2:slope points towards trial,3: maxStep>minStep, else return early
	// reconsider casecode0, MP2 doesn't have
	if (bracketed && ((trialpoint.step <= math.Min(bestpoint.step, otherendpoint.step)) || (trialpoint.step >= math.Max(bestpoint.step, otherendpoint.step)))) || (bestpoint.slope*(trialpoint.step-bestpoint.step) >= 0.0) || (stpmax < stpmin) {
		return bestpoint, otherendpoint, trialpoint.step, bracketed, 0
	}

	// Trial counts as "higher" only if it exceeds best by more than float32 noise.
	// Within noise, the slopes decide (cases 2-4) instead of a random energy ordering.
	higher := trialpoint.f > bestpoint.f+fnoise
	sgnd := trialpoint.slope * (bestpoint.slope / math.Abs(bestpoint.slope))
	bound := false
	var chosenstep, cubicstep, quadstep float64

	switch {
	//trial overshot, so minimum is bracketed between best and trial
	case higher:
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
	if higher {
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

// Termination codes returned by MTlinesearch.
const (
	lsNoDescent  = -1 // slope0 >= 0: nothing evaluated, M and g untouched
	lsRunning    = 0  // internal: keep iterating
	lsConverged  = 1  // strong (or approximate) Wolfe conditions satisfied
	lsBracketTol = 2  // bracket narrower than tolerance
	lsMaxEvals   = 3  // evaluation budget exhausted (returns best point)
	lsAtMinStep  = 4  // stuck at minStep
	lsAtMaxStep  = 5  // stuck at maxStep (includes the max-rotation-angle cap)
	lsNoProgress = 6  // cstep failed, or trial left the bracket
)

// Moré-Thuente line search (Moré & Thuente, ACM TOMS 20(3), 1994; MINPACK-2).
// Along direction s from base point wa, define
//
//	φ(α)  = E(normalize(wa + α s)),    φ'(α) = g(α)·s
//
// and look for α satisfying the strong Wolfe conditions
//
//	φ(α)  ≤ φ(0) + μ α φ'(0)     sufficient decrease,  μ = ftol
//	|φ'(α)| ≤ η |φ'(0)|          curvature,            η = gtol
//
// On entry g holds the gradient at wa and f0 = φ(0). evalEG moves nothing
// itself: this function sets M = wa + step*s, then evalEG normalizes M, writes
// the gradient into g, and returns the energy. On return M, g and the returned
// energy correspond to the returned step. step0 is the first trial step;
// maxStepAngle (degrees, <=0 disables) caps how far any cell's m may rotate.
func MTlinesearch(wa *data.Slice, f0 float64, g *data.Slice, slope0, dirNorm, step0 float64, s *data.Slice,
	evalEG func(*data.Slice) float64, verbose int, maxStepAngle float64) (newF, newStep float64, info int) {

	const (
		// these are all tunable
		ftol         = 1e-4          // μ (Wolfe c1)
		gtol         = 0.9           // η (Wolfe c2); loose, standard for quasi-Newton
		f32eps       = 1.1920929e-07 // float32 eps: energies come from float32 reductions
		extrapFactor = 4.0           // unbracketed: next trial <= step + extrapFactor*(step-best)
		extrapLower  = 1.1           // unbracketed: next trial >= step + extrapLower*(step-best)
		maxEvals     = 20
		bracketTol   = 1e-7 // stop when bracket width <= bracketTol * upper end
		// Hager-Zhang approximate-Wolfe slope bound: φ'(α) ≤ (2μ-1)φ'(0). Note 2μ-1 < 0 here.
		approxSlopeCoef = 2*ftol - 1
	)

	// slope0=g.s is φ'(0); passed in, must be negative (s is a descent direction)
	if slope0 >= 0 {
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch (Wolfe): no descent %e\n", slope0)
		}
		return f0, 0, lsNoDescent
	}

	// Step cap from max rotation angle. s is ~tangent to m, so a step α rotates
	// a cell by atan(α|s|); hence α_max = tan(angle)/max|s|.
	//dirNorm is max|s| over cells, used only for the angle cap
	const maxAngleLimit = 89.0 // degrees; tan stays finite, and 90°+ can't be reached by a tangent step anyway
	// Fallback limits if the direction norm is unknown. With dirNorm known, limit the
	// displacement step*|s| instead of step itself: below ~4 float32 ulps of |m|=1 the
	// move is rounded away, and above ~1e3 displacement the trial point on the sphere has saturated
	minStep := 1e-15 // MT's stpmin: fixed, independent of step0 and dirNorm
	maxStep := 1e15
	if dirNorm > 0 {
		maxStep = 1e3 / dirNorm
	}
	angleCapped := false
	if maxStepAngle > 0 && dirNorm > 0 {
		angle := math.Min(maxStepAngle, maxAngleLimit)
		if capStep := math.Tan(angle*math.Pi/180.0) / dirNorm; capStep < maxStep {
			maxStep = capStep
			angleCapped = true
		}
	}

	maxStep = math.Max(maxStep, minStep) // keep maxStep >= minStep so cstep doesn't see stpmax < stpmin
	// Stage 1 works on ψ(α) = φ(α) − φ(0) − μφ'(0)α, ψ'(α) = φ'(α) − μφ'(0),
	// for which ψ(α) ≤ 0 is exactly sufficient decrease. φ(0) is dropped here.
	armijoSlope := ftol * slope0 // μφ'(0): slope of the sufficient-decrease line φ(0) + armijoSlope·α (negative)
	toPsi := func(p lsPoint) lsPoint { return lsPoint{p.step, p.f - p.step*armijoSlope, p.slope - armijoSlope} }
	fromPsi := func(p lsPoint) lsPoint { return lsPoint{p.step, p.f + p.step*armijoSlope, p.slope + armijoSlope} }

	// Interval of uncertainty [α_l, α_u] (MT notation). bestPoint = α_l, the
	// best point so far; otherEndPoint = α_u. Once bracketed, the interval
	// contains a point satisfying the strong Wolfe conditions.
	bestPoint := lsPoint{0, f0, slope0}
	otherEndPoint := bestPoint

	step := step0
	bracketed, stage1 := false, true
	casecode := 1 // cstep case code; 0 means cstep failed
	numEvals := 0
	width, width1 := maxStep-minStep, 2*(maxStep-minStep) // for the bisection safeguard
	var stmin, stmax float64
	fnoise := f32eps * math.Abs(f0) // float32 resolution of the energy

	for {
		if math.IsNaN(step) || math.IsInf(step, 0) {
			if verbose > 0 {
				fmt.Println("WARNING: linesearch: NaN/Inf step, resetting to minStep")
			}
			step = minStep
		}
		// Limits handed to cstep.
		if bracketed {
			stmin = math.Min(bestPoint.step, otherEndPoint.step)
			stmax = math.Max(bestPoint.step, otherEndPoint.step)
		} else {
			if numEvals == 0 {
				stmin = 0 // no forced growth before the first evaluation
			} else {
				stmin = step + extrapLower*(step-bestPoint.step)
			}
			stmax = step + extrapFactor*(step-bestPoint.step)
		}

		step = math.Min(math.Max(step, minStep), maxStep)
		if verbose > 1 && angleCapped && step == maxStep {
			fmt.Printf("linesearch: step clamped by MaxStepAngle (%.2f deg)\n", maxStepAngle)
		}

		// Out of evals, stalled, or bracket collapsed: fall back to the best
		// point so M and g end up there.
		// TODO(review): zero-step return. If no trial ever improved on the start
		// (bestPoint.step == 0), this fallback evaluates at wa, so M and g are
		// restored to the start point, and the function returns newStep == 0 with
		// info = lsNoProgress / lsMaxEvals / lsAtMinStep. Callers MUST check for
		// step == 0 (or info != lsConverged) and treat it as a failed line search:
		//   - SD (Minimizer.SD_linesearch): mini.h = 0 would make the next BB step
		//     a zero step (cuda.Minimize with h = 0); keep the previous h instead.
		//   - L-BFGS (Step): rate == 0 gives s = 0, so the history update is
		//     skipped and the iteration repeats; reset history / fall back to
		//     steepest descent instead.
		if (bracketed && (step <= stmin || step >= stmax)) || numEvals >= maxEvals-1 ||
			casecode == 0 || (bracketed && stmax-stmin <= bracketTol*stmax) {
			step = bestPoint.step
		}

		cuda.Madd2(M.Buffer(), wa, s, 1.0, float32(step)) // M = wa + step*s
		f := evalEG(g)                                    // normalizes M, fills g, returns E
		numEvals++
		slope := cuda.SlopeAlongLine(g, wa, s, float32(step))

		ftest1 := f0 + step*armijoSlope + fnoise // Armijo bound: φ(0) + μ α φ'(0), plus fnoise slack (MINPACK's ftest)
		// Hager-Zhang approximate Wolfe (SIOPT 2005): accept if f is within float32
		// noise of f0 and the slope has flattened (φ' ≤ (2μ-1)φ'(0)). Rescues steps
		// whose energy decrease is below float32 resolution, where Armijo can't pass.
		approxWolfe := f <= f0+fnoise && approxSlopeCoef*slope0 >= slope // Hager-Zhang approximate Wolfe
		curvatureOK := math.Abs(slope) <= gtol*(-slope0)

		info = lsRunning
		if (bracketed && (step <= stmin || step >= stmax)) || casecode == 0 {
			info = lsNoProgress
		}
		if step == maxStep && f <= ftest1 && slope <= armijoSlope {
			info = lsAtMaxStep
		}
		// Step too small for float32 to resolve (displacement < ~4 ulps of |m|=1) is
		// equivalent to being stuck at stpmin: shrinking further changes nothing.
		unresolvable := dirNorm > 0 && step*dirNorm <= 4*f32eps
		if (step == minStep || unresolvable) && (f > ftest1 || slope >= armijoSlope) {
			info = lsAtMinStep
		}
		if numEvals >= maxEvals {
			info = lsMaxEvals
		}
		if bracketed && stmax-stmin <= bracketTol*stmax {
			info = lsBracketTol
		}
		if (f <= ftest1 && curvatureOK) || (approxWolfe && curvatureOK) {
			info = lsConverged
		}
		if info != lsRunning {
			if verbose > 1 {
				fmt.Printf("linesearch: info=%d evals=%d step=%e\n", info, numEvals, step)
			}
			return f, step, info
		}

		if stage1 && f <= ftest1 && slope >= 0 {
			stage1 = false
		}

		trial := lsPoint{step, f, slope}

		if stage1 && f <= bestPoint.f+fnoise && f > ftest1 {
			// Stage 1: apply the updating algorithm to ψ instead of φ.
			var b, o lsPoint
			b, o, step, bracketed, casecode = cstep(toPsi(bestPoint), toPsi(otherEndPoint), toPsi(trial), bracketed, stmin, stmax, fnoise)
			bestPoint, otherEndPoint = fromPsi(b), fromPsi(o)
		} else {
			bestPoint, otherEndPoint, step, bracketed, casecode = cstep(bestPoint, otherEndPoint, trial, bracketed, stmin, stmax, fnoise)
		}
		// Bisection safeguard: if the bracket hasn't shrunk to <66% of its
		// width two iterations ago, discard cstep's proposal and bisect.
		if bracketed {
			if math.Abs(otherEndPoint.step-bestPoint.step) >= 0.66*width1 {
				step = bestPoint.step + 0.5*(otherEndPoint.step-bestPoint.step)
			}
			width1 = width
			width = math.Abs(otherEndPoint.step - bestPoint.step)
		}
	}
}
