package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

const f32eps = 1.1920929e-07

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

// line-search machinery (strong-Wolfe) shared between LBFGSMinimizer and the steepest-descent Minimizer's inexact-line-search steps.
// implementation largely follows MINPACK-2's dcstep and dcsrch (Moré & Thuente, ACM TOMS 20(3), 1994).

type lsPoint struct {
	step, f, slope float64
}

func cubicCoeffs(fa, da, sa, fb, db, sb float64) (theta, gamma float64) {
	theta = 3.0*(fa-fb)/(sb-sa) + da + db
	s := math.Max(math.Abs(theta), math.Max(math.Abs(da), math.Abs(db)))
	gamma = s * math.Sqrt(math.Max(0.0, (theta/s)*(theta/s)-(da/s)*(db/s)))
	return theta, gamma
}

// Given the two current bracket endpoints and a newly evaluated trial point t, update the bracket and propose a next trial step.
// stpmin/stpmax are NOT a global clamp, they only bound the step while extrapolating (unbracketed cases 3 and 4)
func cstep(bestpoint, otherendpoint, trialpoint lsPoint, bracketed bool, stpmin, stpmax, fnoise float64) (newBestPoint, newOtherEndPoint lsPoint, newStep float64, newIsBracketed bool, casecode int) {

	// check: 1) bracketed, 2) min<step<max, 3) slope points toward trial, 4) stepmax>stepmin
	if (bracketed && ((trialpoint.step <= math.Min(bestpoint.step, otherendpoint.step)) || (trialpoint.step >= math.Max(bestpoint.step, otherendpoint.step)))) || (bestpoint.slope*(trialpoint.step-bestpoint.step) >= 0.0) || (stpmax < stpmin) {
		return bestpoint, otherendpoint, trialpoint.step, bracketed, 0
	}

	higher := trialpoint.f > bestpoint.f+fnoise
	sgnd := trialpoint.slope * (bestpoint.slope / math.Abs(bestpoint.slope))
	var chosenstep, cubicstep, quadstep float64

	switch {
	// Case 1: trial overshot, so the minimum is bracketed between best and trial.
	// Cubic if it is closer to best than the quadratic, else their average.
	case higher:
		casecode = 1
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
			chosenstep = (cubicstep + quadstep) / 2.0
		}
		bracketed = true

	// Case 2: lower f, slopes of opposite sign. Bracketed.
	// Cubic if it is farther from trial than the secant, else the secant (minimizer of the quadratic).
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

	// Case 3: lower f, same-sign slopes, |slope| decreasing.
	case math.Abs(trialpoint.slope) < math.Abs(bestpoint.slope):
		casecode = 3
		theta, gamma := cubicCoeffs(bestpoint.f, bestpoint.slope, bestpoint.step, trialpoint.f, trialpoint.slope, trialpoint.step)
		if trialpoint.step > bestpoint.step {
			gamma = -gamma
		}
		p := (gamma - trialpoint.slope) + theta
		q := (gamma + (bestpoint.slope - trialpoint.slope)) + gamma
		r := p / q
		// Cubic only if it tends to infinity in the step direction or its
		// minimum lies beyond trial; otherwise use the extrapolation limit.
		if (r < 0.0) && (gamma != 0.0) {
			cubicstep = trialpoint.step + r*(bestpoint.step-trialpoint.step)
		} else if trialpoint.step > bestpoint.step {
			cubicstep = stpmax
		} else {
			cubicstep = stpmin
		}
		quadstep = trialpoint.step + (trialpoint.slope/(trialpoint.slope-bestpoint.slope))*(bestpoint.step-trialpoint.step)
		if bracketed {
			// Cubic if closer to trial than the secant, else the secant; then
			// stay within 66% of the way to the other endpoint.
			if math.Abs(cubicstep-trialpoint.step) < math.Abs(quadstep-trialpoint.step) {
				chosenstep = cubicstep
			} else {
				chosenstep = quadstep
			}
			if trialpoint.step > bestpoint.step {
				chosenstep = math.Min(trialpoint.step+0.66*(otherendpoint.step-trialpoint.step), chosenstep)
			} else {
				chosenstep = math.Max(trialpoint.step+0.66*(otherendpoint.step-trialpoint.step), chosenstep)
			}
		} else {
			// Extrapolating: cubic if farther from trial than the secant.
			if math.Abs(cubicstep-trialpoint.step) > math.Abs(quadstep-trialpoint.step) {
				chosenstep = cubicstep
			} else {
				chosenstep = quadstep
			}
			chosenstep = math.Min(stpmax, chosenstep)
			chosenstep = math.Max(stpmin, chosenstep)
		}

	// Case 4: lower f, same-sign slopes, |slope| not decreasing.
	default:
		casecode = 4
		if bracketed {
			theta, gamma := cubicCoeffs(trialpoint.f, trialpoint.slope, trialpoint.step, otherendpoint.f, otherendpoint.slope, otherendpoint.step)
			if trialpoint.step > otherendpoint.step {
				gamma = -gamma
			}
			p := (gamma - trialpoint.slope) + theta
			q := ((gamma - trialpoint.slope) + gamma) + otherendpoint.slope
			r := p / q
			chosenstep = trialpoint.step + r*(otherendpoint.step-trialpoint.step)
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

	return bestpoint, otherendpoint, chosenstep, bracketed, casecode
}

// Moré-Thuente line search (Moré & Thuente, ACM TOMS 20(3), 1994; MINPACK-2).
// Along direction s from base point m0, define
//
//	φ(α)  = E(normalize(m0 + α s)),    φ'(α) = g(α)·s
//
// and look for α satisfying the strong Wolfe conditions
//
//	φ(α)  ≤ φ(0) + μ α φ'(0)     sufficient decrease,  μ = ftol
//	|φ'(α)| ≤ η |φ'(0)|          curvature,            η = gtol
//
// On entry g holds the gradient at m0 and f0 = φ(0). evalEG moves nothing
// itself: this function sets M = m0 + step*s, then evalEG normalizes M, writes
// the gradient into g, and returns the energy. On return M, g and the returned
// energy correspond to the returned step. step0 is the first trial step;
// maxStepAngle (degrees, <=0 disables) caps how far any cell's m may rotate.
func MTlinesearch(m0 *data.Slice, f0 float64, g *data.Slice, maxDirNorm, step0 float64, s *data.Slice,
	evalEG func(*data.Slice) float64, verbose int, maxStepAngle float64) (newF, newStep float64, info int) {

	const (
		// these are all tunable
		ftol         = 1e-4 // μ (Wolfe c1)
		gtol         = 0.9  // η (Wolfe c2); loose, standard for quasi-Newton
		extrapFactor = 4.0  // unbracketed: next trial <= step + extrapFactor*(step-best)
		extrapLower  = 1.1  // unbracketed: next trial >= step + extrapLower*(step-best)
		maxEvals     = 20
		bracketTol   = 1e-7 // stop when bracket width <= bracketTol * upper end, floored at 4*f32eps
		// Hager-Zhang approximate-Wolfe slope bound: φ'(α) ≤ (2μ-1)φ'(0). Note 2μ-1 < 0 here.
		//approxSlopeCoef = 2*ftol - 1
	)

	if !(step0 > 0) || math.IsInf(step0, 0) { // also catches NaN
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch: invalid step0 %e\n", step0)
		}
		return f0, 0, lsNoProgress
	}

	ms := Msat.MSlice()
	defer ms.Recycle()
	V := cellVolume() // applied on the host in float64; keeps ~1e-26 out of float32

	// slope0= Σ V·M_s·g·s at step 0 (|m0| = 1, so the 1/|x| factor is 1)
	// slope0= g.s is φ'(0); must be negative (s is a descent direction)
	slope0 := V * cuda.SlopeAlongLine(g, m0, s, ms, 0)
	if slope0 >= 0 {
		if verbose > 0 {
			fmt.Printf("WARNING: linesearch (Wolfe): no descent %e\n", slope0)
		}
		return f0, 0, lsNoDescent
	}
	if verbose > 1 {
		fmt.Printf("ls entry: f0=%e slope0=%e step0=%e maxDirNorm=%e\n", f0, slope0, step0, maxDirNorm)
	}

	// Step cap from max rotation angle. maxDirNorm is max|s| over cells
	// s is ~tangent to m, so a step α rotates a cell by atan(α|s|); hence α_max = tan(angle)/max|s|.
	// Fallback limits if the direction norm is unknown. With maxDirNorm known, limit the
	// displacement step*|s| instead of step itself: below ~4 float32 ulps of |m|=1 the
	// move is rounded away, and above ~1e3 displacement the trial point on the sphere has saturated
	// MT suggest α_max = (φ_min − φ(0)) / (μ φ'(0)) for φ bounded below, unused since our displacement/angle caps are stricter.
	minStep := 1e-15 // MT's stpmin: fixed, independent of step0 and maxDirNorm
	maxStep := 1e15
	if maxDirNorm > 0 {
		maxStep = 1e3 / maxDirNorm
	}

	// Angle cap. s is ~tangent to m, so a step α rotates a cell by atan(α|s|);
	// hence α_max = tan(angle)/max|s|. maxAngleLimit keeps tan finite
	// TODO(revisit): the cap formula changes under a sphere-curve search (see note above)
	const maxAngleLimit = 89.0
	effAngle := 0.0 // effective angle cap in degrees; 0 means no angle cap
	angleCapped := false
	if maxStepAngle > 0 && maxDirNorm > 0 {
		effAngle = math.Min(maxStepAngle, maxAngleLimit)
		if capStep := math.Tan(effAngle*math.Pi/180.0) / maxDirNorm; capStep < maxStep {
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
	casecode := 1 // cstep case code; 0 means no usuable cstep step, fallsback to MTlinesearch's bestPoint
	numEvals := 0
	width, width1 := maxStep-minStep, 2*(maxStep-minStep) // for the bisection safeguard
	var stmin, stmax float64
	//replace | f0 | with max(|f0|, Σ|term energies|), recorded as LastEnergyScale inside GetTotalEnergy().
	fnoise := f32eps * math.Abs(f0) // float32 resolution of the energy

	for {
		if math.IsNaN(step) || math.IsInf(step, 0) {
			if verbose > 0 {
				fmt.Println("WARNING: linesearch: NaN/Inf step, fallback to beststep")
			}
			casecode = 0
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
		// Bracket too narrow to subdivide. Two scales: relative to the step
		// (dcsrch's xtol test), and absolute displacement in m, since float32
		// can't resolve a move below ~4 ulps of |m| = 1.
		bracketTight := bracketed && (stmax-stmin <= bracketTol*stmax ||
			(maxDirNorm > 0 && (stmax-stmin)*maxDirNorm <= 4*f32eps))

		// warning: step stays float64 and must not be overwritten with a
		// float32-rounded value after this line: the step == maxStep / step == minStep tests
		// in the info block rely on the clamp assigning the limits exactly. The float32 cast
		// happens only at the use sites (Madd2, SlopeAlongLine).
		step = math.Min(math.Max(step, minStep), maxStep)
		if verbose > 1 && angleCapped && step == maxStep {
			fmt.Printf("linesearch: step clamped by angle cap (%.2f deg)\n", effAngle)
		}

		// Out of evals, stalled, or bracket collapsed: fall back to the best
		// point so M and g end up there.
		// TODO(review): zero-step return. If no trial ever improved on the start
		// (bestPoint.step == 0), this fallback evaluates at m0, so M and g are
		// restored to the start point, and the function returns newStep == 0 with
		// info = lsNoProgress / lsMaxEvals / lsAtMinStep/ lsBracketTol. Callers MUST check for
		// step == 0 (or info != lsConverged) and treat it as a failed line search:
		//   - SD (Minimizer.SD_linesearch): mini.h = 0 would make the next BB step
		//     a zero step (cuda.Minimize with h = 0); keep the previous h instead.
		//   - L-BFGS (Step): rate == 0 gives s = 0, so the history update is
		//     skipped and the iteration repeats; reset history / fall back to
		//     steepest descent instead.
		if (bracketed && (step <= stmin || step >= stmax)) || numEvals >= maxEvals-1 ||
			casecode == 0 || bracketTight {
			step = bestPoint.step
		}

		// TODO(revisit): search along a curve on the sphere instead of normalize(base + α·dir).
		//
		// Currently the trial point is the straight line x(α) = base + α·dir, projected
		// back with normalize(). An alternative is a rotation-like curve that stays on
		// the sphere by construction, e.g. the Cayley form used by cuda/minimize.cu:
		//
		//	m(α) = ((4 - α²|t|²)·m0 + 4α·t) / (4 + α²|t|²),   t tangent to m0
		//
		// This is an exact rotation by 2·atan(α|t|/2). It agrees with the current curve
		// to first order, so φ'(0) = g·t is unchanged, but the curves diverge at large α.
		// Candidate slope (derive and FD-check before trusting; uses g·m = 0 at the trial point):
		//
		//	φ'(α) = V·Σ Ms·(4 g·t - 2α|t|² g·m0) / (4 + α²|t|²)
		//
		// It has the same inputs as the current slope kernel, drops rsqrtf and |x|, and
		// adds one dot product.
		//
		// What it would improve:
		//
		// SD (Minimizer):
		//   - Its BB step already moves along the Cayley curve via cuda.Minimize(m, m0, k, h),
		//     but this line search optimizes along normalize(m0 + α·k). The accepted step h is
		//     therefore applied on a slightly different curve than the one it was tuned on.
		//     Using cuda.Minimize as the trial-point generator removes that mismatch, and the
		//     torque k is exactly tangent, so no projection is needed.
		//   - Cleaner angle cap: rotation is exactly 2·atan(α|t|/2), so
		//     α_max = 2·tan(θmax/2)/maxDirNorm (≈1.97/maxDirNorm at 89°) instead of
		//     tan(θmax)/maxDirNorm (≈57/maxDirNorm at 89°).
		//   - BB seeds are already in Cayley units, so they stay consistent with the search.
		//
		// L-BFGS:
		//   - No |x| factor, no lost-norm issue: M is on the sphere at every trial.
		//   - Same cleaner angle cap as above.
		//   - Requires a tangent direction. The two-loop recursion mixes history vectors from
		//     other points, so dir is generally NOT tangent to base. Project once per
		//     iteration, dir_t = dir - (dir·m)·m, before the search. This leaves φ'(0)
		//     unchanged since g·m = 0 implies g·dir_t = g·dir. Without projection |m| != 1
		//     and the slope formula above is wrong. Needs a new trial-point kernel (SD can
		//     reuse minimize).
		//   - Keep M.normalize() in evalEG as float32 insurance (becomes a near no-op).
		//
		// Related, larger design question (separate change): vector transport of the L-BFGS
		// history. The stored s_i and y_i live in tangent spaces at different points of the
		// sphere, and the two-loop recursion combines them as if they shared one. The
		// simplest transport is to project each stored vector onto the current tangent space
		// (v -> v - (v·m)·m) before use, or at insertion time. Related: y = grad - grad_old
		// mixes gradients from different tangent spaces, and s = M - x_old is a chord, not a
		// tangent vector. Also, the history uses the unweighted g, not the V·Ms-weighted
		// gradient, so it is not a quasi-Newton model of f (see the weighted-metric item).
		//
		// Suggested order: do SD first (nearly free, removes a real inconsistency), measure
		// iteration counts before/after in isolation, then L-BFGS, then consider transport.
		// Not done yet because the current formulation is validated (FD/slope ≈ 1.000) and
		// this touches the kernel, both callers and the cap logic together.
		cuda.Madd2(M.Buffer(), m0, s, 1.0, float32(step)) // M = m0 + step*s
		f := evalEG(g)                                    // normalizes M, fills g, returns E
		numEvals++
		slope := V * cuda.SlopeAlongLine(g, m0, s, ms, float32(step))

		if verbose > 1 {
			fmt.Printf("ls trial %d: step=%e dE=%e FD=%e slope=%e FD/slope=%e\n",
				numEvals, step, f-f0, (f-f0)/step, slope, ((f-f0)/step)/slope)
		}

		ftest1 := f0 + step*armijoSlope + fnoise // Armijo bound: φ(0) + μ α φ'(0), plus fnoise slack (MINPACK's ftest)
		// Hager-Zhang approximate Wolfe (SIOPT 2005): accept if f is within float32
		// noise of f0 and the slope has flattened (φ' ≤ (2μ-1)φ'(0)). Rescues steps
		// whose energy decrease is below float32 resolution, where Armijo can't pass.
		//approxWolfe := f <= f0+fnoise && approxSlopeCoef*slope0 >= slope // Hager-Zhang approximate Wolfe
		//^-- double check if HZ is needed, fnoise might already handle this
		curvatureOK := math.Abs(slope) <= gtol*(-slope0)

		info = lsRunning

		// bracketed but hit edge of bracket or outside
		if (bracketed && (step <= stmin || step >= stmax)) || casecode == 0 {
			info = lsNoProgress
		}

		// hit maxStep but it's still decreasing sufficiently (and so is slope)
		if step == maxStep && f <= ftest1 && slope <= armijoSlope {
			info = lsAtMaxStep
		}
		// hit minstep, or equivalently Step too small for float32 to resolve (displacement < ~4 ulps of |m|=1) is
		// equivalent to being stuck at stpmin: shrinking further changes nothing.
		unresolvable := maxDirNorm > 0 && step*maxDirNorm <= 4*f32eps
		if (step == minStep || unresolvable) && (f > ftest1 || slope >= armijoSlope) {
			info = lsAtMinStep
		}
		if numEvals >= maxEvals {
			info = lsMaxEvals
		}
		//bracket shrank too small
		if bracketTight {
			info = lsBracketTol
		}
		//passes both strong Wolfe conditions
		//if (f <= ftest1 && curvatureOK) || (approxWolfe && curvatureOK) {
		if f <= ftest1 && curvatureOK {
			info = lsConverged
		}
		if info != lsRunning {
			if verbose > 1 {
				fmt.Printf("linesearch: info=%d evals=%d step=%e\n", info, numEvals, step)
				if info == lsBracketTol {
					fmt.Printf("linesearch: bracket displacement=%e (float32 floor %e)\n",
						(stmax-stmin)*maxDirNorm, 4*f32eps)
				}
			}
			return f, step, info
		}

		if stage1 && f <= ftest1 && slope >= 0 {
			stage1 = false
		}

		trial := lsPoint{step, f, slope}
		// if stage1, f is better than bestp, and psi>0, bracket on psi
		if stage1 && f <= bestPoint.f+fnoise && f > ftest1 {
			// Stage 1: apply the updating algorithm to ψ instead of φ.
			var b, o lsPoint
			b, o, step, bracketed, casecode = cstep(toPsi(bestPoint), toPsi(otherEndPoint), toPsi(trial), bracketed, stmin, stmax, fnoise)
			bestPoint, otherEndPoint = fromPsi(b), fromPsi(o)
		} else {
			// Work on φ itself: stage 2, or stage 1 when the trial is worse than best/trial satisfies Armijo
			bestPoint, otherEndPoint, step, bracketed, casecode = cstep(bestPoint, otherEndPoint, trial, bracketed, stmin, stmax, fnoise)
		}
		// Bisection safeguard: if the bracket hasn't shrunk to <66% of its
		// width two iterations ago, discard cstep's proposal and bisect. new trial point is midway between bracket endpoints
		if bracketed {
			if math.Abs(otherEndPoint.step-bestPoint.step) >= 0.66*width1 {
				step = bestPoint.step + 0.5*(otherEndPoint.step-bestPoint.step)
			}
			width1 = width
			width = math.Abs(otherEndPoint.step - bestPoint.step)
		}
	}
}
