package engine

// Riemannian trust-region minimizer with Steihaug-Toint truncated CG.
//
// Conventions (all vectors are tangent to the unit sphere in every cell):
//   t  = torque from torqueFn = P_m B_eff = -grad E   (Precess is off during Minimize)
//   g  = -t,   H = Riemannian Hessian,   model q(s) = <g,s> + 1/2 <s,Hs>
// H*v is obtained by a finite difference of the torque along the retraction (one torqueFn
// call per product), see hessVec. Swap that function for an exact product later if wanted.
//
// The actual energy reduction is NOT computed from the (float32, ~1e-7 relative) total energy.
// It is obtained by integrating <t, gamma'> along the retraction path with the trapezoid
// rule, using the torque at both ends (which we need anyway). This is exact for a quadratic
// energy and has no cancellation noise near convergence.
//
// Assumes uniform Msat (torque is then a constant multiple of the gradient in the plain
// Euclidean metric). With region-dependent Msat the dot products need cell weights.

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MinimizerTR    bool    = false // use trust-region minimizer instead of steepest descent
	TRInitRadius   float64 = 0.1   // initial trust radius, as RMS rotation angle per cell (rad)
	TRMaxRadius    float64 = 1.0   // maximum trust radius, as RMS rotation angle per cell (rad)
	TRMaxCG        int     = 20    // max inner CG iterations (one torque evaluation each)
	TRFDAngle      float64 = 1e-2  // finite-difference probe: largest per-cell rotation (rad)
	TRMaxStepAngle float64 = 1.0   // cap on the largest per-cell step |s_i| (~rad)
)

const (
	trEta       = 0.05 // accept step if rho > trEta
	trMaxReject = 20   // consecutive rejections after which the step counts as "no progress"
)

func init() {
	DeclVar("MinimizerTR", &MinimizerTR, "Use the Riemannian trust-region minimizer (truncated CG) in Minimize() instead of steepest descent.")
	DeclVar("TRInitRadius", &TRInitRadius, "Initial trust radius of the TR minimizer, as RMS rotation angle per cell (rad).")
	DeclVar("TRMaxRadius", &TRMaxRadius, "Maximum trust radius of the TR minimizer, as RMS rotation angle per cell (rad).")
	DeclVar("TRMaxCG", &TRMaxCG, "Maximum inner CG iterations per TR step.")
	DeclVar("TRFDAngle", &TRFDAngle, "Largest per-cell rotation (rad) used in the finite-difference Hessian-vector product.")
	DeclVar("TRMaxStepAngle", &TRMaxStepAngle, "Cap on the largest per-cell rotation (rad) of a TR step.")
}

type trMinimizer struct {
	t, tmp  *data.Slice // torque at current m; scratch torque (trial point / FD probe)
	tmp2    *data.Slice // second scratch torque, only allocated for central differences
	avoid   *data.Slice // optional: unit tangent vector that steps must stay orthogonal to (not owned, see escape.go)
	zero    *data.Slice // zero vector, only allocated when avoid is used
	m0      *data.Slice // magnetization at start of step
	s, r, d *data.Slice // tCG: step, residual (g + Hs), search direction
	hd      *data.Slice // H*d, later scratch
	delta   float64     // trust radius, 2-norm over all cells
	gnorm0  float64     // |g| at start, scale reference for the CG tolerance
	nreject int
	lastDm  fifoRing
}

func trDot(a, b *data.Slice) float64 { return float64(cuda.Dot(a, b)) }

func (tr *trMinimizer) alloc(size [3]int) {
	tr.t = cuda.Buffer(3, size)
	tr.tmp = cuda.Buffer(3, size)
	tr.m0 = cuda.Buffer(3, size)
	tr.s = cuda.Buffer(3, size)
	tr.r = cuda.Buffer(3, size)
	tr.d = cuda.Buffer(3, size)
	tr.hd = cuda.Buffer(3, size)
}

func (tr *trMinimizer) Free() {
	for _, b := range []**data.Slice{&tr.t, &tr.tmp, &tr.tmp2, &tr.zero, &tr.m0, &tr.s, &tr.r, &tr.d, &tr.hd} {
		if *b != nil {
			cuda.Recycle(*b)
			*b = nil
		}
	}
}

// hd = Hess[d] at the current m (tr.m0 must hold m, tr.t the torque at m).
// Forward finite difference of the torque along the (second-order) retraction.
func (tr *trMinimizer) hessVec(hd, d *data.Slice) {
	dmax := cuda.MaxVecNorm(d)
	if dmax == 0 {
		cuda.Madd2(hd, d, d, 0, 0)
		return
	}
	m := M.Buffer()
	eps := float32(TRFDAngle / dmax)

	cuda.Minimize(m, tr.m0, d, eps) // m := R_m0(eps*d)
	torqueFn(tr.tmp)
	data.Copy(m, tr.m0) // restore

	cuda.TRHessVec(hd, tr.tmp, tr.t, tr.m0, 1/eps)
}

// Same as hessVec but with a central difference (O(eps^2) instead of O(eps) truncation error,
// two torque evaluations). Only needs tr.m0 (= current m); used for the eigenvalue analysis.
func (tr *trMinimizer) hessVecCentral(hd, d *data.Slice) {
	dmax := cuda.MaxVecNorm(d)
	if dmax == 0 {
		cuda.Madd2(hd, d, d, 0, 0)
		return
	}
	m := M.Buffer()
	if tr.tmp2 == nil {
		tr.tmp2 = cuda.Buffer(3, m.Size())
	}
	eps := float32(TRFDAngle / dmax)

	cuda.Minimize(m, tr.m0, d, eps)
	torqueFn(tr.tmp)
	cuda.Minimize(m, tr.m0, d, -eps)
	torqueFn(tr.tmp2)
	data.Copy(m, tr.m0) // restore

	cuda.TRHessVec(hd, tr.tmp, tr.tmp2, tr.m0, 0.5/eps)
}

// Steihaug-Toint truncated CG for min <g,s> + 1/2 <s,Hs> s.t. |s| <= delta.
// On return tr.s is the step and tr.r = g + H s. Returns true if the step ended on the boundary.
func (tr *trMinimizer) tCG(gnorm float64) (boundary bool) {
	cuda.Madd2(tr.s, tr.t, tr.t, 0, 0)  // s = 0
	cuda.Madd2(tr.r, tr.t, tr.t, -1, 0) // r = g = -t
	if tr.avoid != nil {                // hyperplane-constrained: work with the transverse gradient
		tr.removeAvoid(tr.r)
		gnorm = math.Sqrt(trDot(tr.r, tr.r))
		if gnorm == 0 {
			return false
		}
	}
	cuda.Madd2(tr.d, tr.r, tr.r, -1, 0) // d = -r

	rel := math.Max(1e-3, math.Min(0.1, gnorm/tr.gnorm0)) // superlinear forcing, floored by FD accuracy
	tol2 := (rel * gnorm) * (rel * gnorm)
	delta2 := tr.delta * tr.delta
	rr := gnorm * gnorm

	for j := 0; j < TRMaxCG; j++ {
		tr.hessVec(tr.hd, tr.d)
		if tr.avoid != nil {
			tr.removeAvoid(tr.hd)
		}
		dHd := trDot(tr.d, tr.hd)
		dd := trDot(tr.d, tr.d)
		ss := trDot(tr.s, tr.s)
		sd := trDot(tr.s, tr.d)
		if dd == 0 {
			return false
		}

		alpha := 0.0
		if dHd > 0 {
			alpha = rr / dHd
		}
		// negative curvature, or the full CG step leaves the region: go to the boundary
		if dHd <= 0 || ss+2*alpha*sd+alpha*alpha*dd >= delta2 {
			disc := sd*sd + dd*(delta2-ss)
			if disc < 0 {
				disc = 0
			}
			tau := (-sd + math.Sqrt(disc)) / dd
			cuda.Madd2(tr.s, tr.s, tr.d, 1, float32(tau))
			cuda.Madd2(tr.r, tr.r, tr.hd, 1, float32(tau))
			return true
		}

		cuda.Madd2(tr.s, tr.s, tr.d, 1, float32(alpha))
		cuda.Madd2(tr.r, tr.r, tr.hd, 1, float32(alpha))
		rrNew := trDot(tr.r, tr.r)
		if rrNew <= tol2 {
			return false
		}
		beta := rrNew / rr
		cuda.Madd2(tr.d, tr.r, tr.d, -1, float32(beta)) // d = -r + beta d
		rr = rrNew
	}
	return false
}

// x -= (avoid.x) avoid
func (tr *trMinimizer) removeAvoid(x *data.Slice) {
	c := trDot(tr.avoid, x)
	cuda.Madd2(x, x, tr.avoid, 1, float32(-c))
}

// make tr.avoid tangent to the current m (tr.m0) again and renormalize it
func (tr *trMinimizer) reprojectAvoid() {
	size := M.Buffer().Size()
	if tr.zero == nil {
		tr.zero = cuda.Buffer(3, size)
		data.Copy(tr.zero, data.NewSlice(3, size)) // host zeros
	}
	cuda.TRHessVec(tr.tmp, tr.avoid, tr.zero, tr.m0, 1) // tmp = (m.a)m - a = -P_m a
	if nrm := math.Sqrt(trDot(tr.tmp, tr.tmp)); nrm > 0 {
		cuda.Madd2(tr.avoid, tr.tmp, tr.tmp, float32(-1/nrm), 0)
	}
}

// recompute the cached torque after M was changed from outside
func (tr *trMinimizer) refreshTorque() {
	if tr.t != nil {
		torqueFn(tr.t)
	}
}

func (tr *trMinimizer) noteReject() {
	tr.nreject++
	if tr.nreject > trMaxReject {
		// radius has collapsed: no further progress possible at this precision
		tr.lastDm.Add(0)
		setLastErr(tr.lastDm.Max())
	}
}

func (tr *trMinimizer) Step() {
	m := M.Buffer()
	sqrtN := math.Sqrt(float64(m.Len()))

	if tr.t == nil {
		tr.alloc(m.Size())
		torqueFn(tr.t)
		tr.delta = TRInitRadius * sqrtN
		tr.gnorm0 = math.Sqrt(trDot(tr.t, tr.t))
	}
	deltaMax := TRMaxRadius * sqrtN
	if tr.delta > deltaMax {
		tr.delta = deltaMax
	}

	gnorm := math.Sqrt(trDot(tr.t, tr.t))
	if gnorm == 0 { // exactly stationary
		tr.lastDm.Add(0)
		setLastErr(tr.lastDm.Max())
		NSteps++
		return
	}

	data.Copy(tr.m0, m)
	if tr.avoid != nil {
		tr.reprojectAvoid()
	}
	boundary := tr.tCG(gnorm)

	ts0 := trDot(tr.t, tr.s) // <t,s> = -<g,s>
	sr := trDot(tr.s, tr.r)  // <s, g+Hs>
	ss := trDot(tr.s, tr.s)
	sHs := sr + ts0 // <s,Hs> = <s,r> - <g,s>
	_ = ss

	// per-cell angle cap, applied as a uniform scaling of the step
	c := 1.0
	if maxs := cuda.MaxVecNorm(tr.s); maxs > TRMaxStepAngle {
		c = TRMaxStepAngle / maxs
		boundary = true
	}
	pred := c*ts0 - 0.5*c*c*sHs // model reduction of c*s

	if !(pred > 0) {
		tr.delta *= 0.25
		tr.noteReject()
		NSteps++
		return
	}

	// trial point
	cuda.Minimize(m, tr.m0, tr.s, float32(c))
	M.normalize()
	torqueFn(tr.tmp)

	// actual reduction: trapezoid rule for int <t, gamma'> dtau along the retraction path
	cuda.TREndVel(tr.hd, tr.m0, tr.s, float32(c))
	v1 := trDot(tr.tmp, tr.hd)
	actual := 0.5 * (c*ts0 + v1)
	rho := actual / pred
	if math.IsNaN(rho) || math.IsInf(rho, 0) {
		rho = -1
	}

	// radius update
	if rho < 0.25 {
		tr.delta *= 0.25
	} else if rho > 0.75 && boundary {
		tr.delta = math.Min(2*tr.delta, deltaMax)
	}

	if rho > trEta {
		tr.t, tr.tmp = tr.tmp, tr.t // new torque becomes current

		cuda.Madd2(tr.hd, m, tr.m0, 1, -1)
		tr.lastDm.Add(cuda.MaxVecNorm(tr.hd))
		setLastErr(tr.lastDm.Max())
		setMaxTorque(tr.t)
		tr.nreject = 0
	} else {
		data.Copy(m, tr.m0) // reject: restore m, tr.t is still valid
		tr.noteReject()
	}

	// as a convention, time does not advance during relax
	NSteps++
}
