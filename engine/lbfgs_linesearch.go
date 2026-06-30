package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func (m *LBFGSMinimizer) lineSearch(
	f *float64,
	searchDir *data.Slice,
	tol float64,
) float32 {

	step := float32(1.0)

	cvsrch(
		m,
		f,
		searchDir,
		&step,
		tol,
	)

	return step
}

func cvsrch(
	m *LBFGSMinimizer,
	f *float64,
	searchDir *data.Slice,
	step *float32,
	tolf float64,
) int {

	const (
		xtol   = 1e-15
		ftol   = 1e-4
		gtol   = 0.9
		stpmin = 1e-15
		stpmax = 1e15
		xtrapf = 4.0
		maxfev = 20
	)

	var (
		info   = 0
		infoc  = 1
		nfev   = 0
		brackt = false
		stage1 = true
	)

	dginit := float64(dot(m.g, searchDir))

	if dginit >= 0 {
		return -1
	}

	finit := *f

	dgtest := ftol * dginit
	eps := tolf

	width := stpmax - stpmin
	width1 := 2 * width

	stx := 0.0
	fx := finit
	dgx := dginit

	sty := 0.0
	fy := finit
	dgy := dginit

	var stmin float64
	var stmax float64

	for {

		// Compute the interval of uncertainty.
		if brackt {
			stmin = math.Min(stx, sty)
			stmax = math.Max(stx, sty)
		} else {
			stmin = stx
			stmax = float64(*step) + xtrapf*(float64(*step)-stx)
		}

		// Clamp step.
		if float64(*step) < stpmin {
			*step = float32(stpmin)
		}
		if float64(*step) > stpmax {
			*step = float32(stpmax)
		}

		// If something went wrong, fall back to the best known step.
		if (brackt && (float64(*step) <= stmin || float64(*step) >= stmax)) ||
			nfev >= maxfev-1 ||
			infoc == 0 ||
			(brackt && (stmax-stmin <= xtol*stmax)) {

			*step = float32(stx)
		}

		// m = x_old + step * searchDir
		cuda.Madd2(
			M.Buffer(),
			m.xOld,
			searchDir,
			1,
			*step,
		)

		M.normalize()

		*f = m.EnergyAndGradient(m.g)

		nfev++

		dg := float64(dot(m.g, searchDir))

		ftest1 := finit + float64(*step)*dgtest
		ftest2 := finit + eps*math.Abs(finit)
		ft := 2*ftol - 1

		info = 0

		if (brackt && (float64(*step) <= stmin || float64(*step) >= stmax)) || infoc == 0 {
			info = 6
		}

		if float64(*step) == stpmax && *f <= ftest2 && dg <= dgtest {
			info = 5
		}

		if float64(*step) == stpmin && (*f > ftest2 || dg >= dgtest) {
			info = 4
		}

		if nfev >= maxfev {
			info = 3
		}

		if brackt && (stmax-stmin <= xtol*stmax) {
			info = 2
		}

		if *f <= ftest1 && math.Abs(dg) <= gtol*(-dginit) {
			info = 1
		}

		if (*f <= ftest2 && ft*dginit >= dg) &&
			(math.Abs(dg) <= gtol*(-dginit)) {
			info = 1
		}

		if info != 0 {
			return info
		}

		if stage1 &&
			(*f <= ftest2 && ft*dginit >= dg) &&
			(dg >= math.Min(ftol, gtol)*dginit) {

			stage1 = false
		}

		if stage1 &&
			(*f <= fx) &&
			!((*f <= ftest2) && (ft*dginit >= dg)) {

			fm := *f - float64(*step)*dgtest
			fxm := fx - stx*dgtest
			fym := fy - sty*dgtest

			dgm := dg - dgtest
			dgxm := dgx - dgtest
			dgym := dgy - dgtest

			cstep(
				&stx, &fxm, &dgxm,
				&sty, &fym, &dgym,
				step,
				&fm, &dgm,
				&brackt,
				stmin,
				stmax,
				&infoc,
			)

			fx = fxm + stx*dgtest
			fy = fym + sty*dgtest
			dgx = dgxm + dgtest
			dgy = dgym + dgtest

		} else {

			cstep(
				&stx, &fx, &dgx,
				&sty, &fy, &dgy,
				step,
				f,
				&dg,
				&brackt,
				stmin,
				stmax,
				&infoc,
			)
		}

		if brackt {

			if math.Abs(sty-stx) >= 0.66*width1 {
				*step = float32(stx + 0.5*(sty-stx))
			}

			width1 = width
			width = math.Abs(sty - stx)
		}
	}

}

func cstep(
	stx, fx, dx *float64,
	sty, fy, dy *float64,
	stp *float32,
	fp, dp *float64,
	brackt *bool,
	stpmin, stpmax float64,
	info *int,
) int {

	*info = 0
	bound := false

	step := float64(*stp)

	// Check input parameters.
	if (*brackt && (step <= math.Min(*stx, *sty) || step >= math.Max(*stx, *sty))) ||
		((*dx)*(step-*stx) >= 0.0) ||
		(stpmax < stpmin) {
		return -1
	}

	sgnd := (*dp) * ((*dx) / math.Abs(*dx))

	var (
		stpf float64
		stpc float64
		stpq float64
	)

	if *fp > *fx {

		*info = 1
		bound = true

		theta := 3.0*((*fx)-(*fp))/(step-(*stx)) + (*dx) + (*dp)
		s := math.Max(theta, math.Max(*dx, *dp))
		gamma := s * math.Sqrt((theta/s)*(theta/s)-((*dx)/s)*((*dp)/s))

		if step < *stx {
			gamma = -gamma
		}

		p := (gamma - (*dx)) + theta
		q := ((gamma - (*dx)) + gamma) + (*dp)
		r := p / q

		stpc = *stx + r*(step-*stx)

		stpq = *stx + ((*dx)/(((*fx)-(*fp))/(step-*stx)+(*dx))/2.0)*(step-*stx)

		if math.Abs(stpc-*stx) < math.Abs(stpq-*stx) {
			stpf = stpc
		} else {
			stpf = stpc + (stpq-stpc)/2.0
		}

		*brackt = true

	} else if sgnd < 0.0 {

		*info = 2
		bound = false

		theta := 3.0*((*fx)-(*fp))/(step-(*stx)) + (*dx) + (*dp)
		s := math.Max(theta, math.Max(*dx, *dp))
		gamma := s * math.Sqrt((theta/s)*(theta/s)-((*dx)/s)*((*dp)/s))

		if step > *stx {
			gamma = -gamma
		}

		p := (gamma - (*dp)) + theta
		q := ((gamma - (*dp)) + gamma) + (*dx)
		r := p / q

		stpc = step + r*((*stx)-step)
		stpq = step + ((*dp)/((*dp)-(*dx)))*((*stx)-step)

		if math.Abs(stpc-step) > math.Abs(stpq-step) {
			stpf = stpc
		} else {
			stpf = stpq
		}

		*brackt = true

	} else if math.Abs(*dp) < math.Abs(*dx) {

		*info = 3
		bound = true

		theta := 3.0*((*fx)-(*fp))/(step-(*stx)) + (*dx) + (*dp)
		s := math.Max(theta, math.Max(*dx, *dp))

		gamma := s * math.Sqrt(math.Max(0.0,
			(theta/s)*(theta/s)-((*dx)/s)*((*dp)/s)))

		if step > *stx {
			gamma = -gamma
		}

		p := (gamma - (*dp)) + theta
		q := (gamma + ((*dx) - (*dp))) + gamma
		r := p / q

		if r < 0.0 && gamma != 0.0 {
			stpc = step + r*((*stx)-step)
		} else if step > *stx {
			stpc = stpmax
		} else {
			stpc = stpmin
		}

		stpq = step + ((*dp)/((*dp)-(*dx)))*((*stx)-step)

		if *brackt {
			if math.Abs(step-stpc) < math.Abs(step-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		} else {
			if math.Abs(step-stpc) > math.Abs(step-stpq) {
				stpf = stpc
			} else {
				stpf = stpq
			}
		}

	} else {

		*info = 4
		bound = false

		if *brackt {

			theta := 3.0*((*fp)-(*fy))/((*sty)-step) + (*dy) + (*dp)
			s := math.Max(theta, math.Max(*dy, *dp))

			gamma := s * math.Sqrt((theta/s)*(theta/s)-((*dy)/s)*((*dp)/s))

			if step > *sty {
				gamma = -gamma
			}

			p := (gamma - (*dp)) + theta
			q := ((gamma - (*dp)) + gamma) + (*dy)
			r := p / q

			stpc = step + r*((*sty)-step)
			stpf = stpc

		} else if step > *stx {

			stpf = stpmax

		} else {

			stpf = stpmin
		}
	}

	if *fp > *fx {

		*sty = step
		*fy = *fp
		*dy = *dp

	} else {

		if sgnd < 0.0 {

			*sty = *stx
			*fy = *fx
			*dy = *dx
		}

		*stx = step
		*fx = *fp
		*dx = *dp
	}

	stpf = math.Min(stpmax, stpf)
	stpf = math.Max(stpmin, stpf)

	step = stpf

	if *brackt && bound {

		if *sty > *stx {
			step = math.Min(*stx+0.66*(*sty-*stx), step)
		} else {
			step = math.Max(*stx+0.66*(*sty-*stx), step)
		}
	}

	*stp = float32(step)

	return 0
}

func (m *LBFGSMinimizer) EnergyAndGradient(g *data.Slice) float64 {

	// g temporarily stores Heff
	SetEffectiveField(g)

	energy := GetTotalEnergy()

	// Convert Heff -> projected energy gradient:
	//
	// g = -(m × (m × Heff))
	//
	// LLNoPrecess already computes exactly
	//
	//      -m × (m × H)
	//
	// which is the downhill direction.

	cuda.LLNoPrecess(
		g,
		M.Buffer(),
		g,
	)

	return energy
}

func (m *LBFGSMinimizer) Free() {

	if m.g != nil {
		m.g.Free()
	}

	if m.gOld != nil {
		m.gOld.Free()
	}

	if m.q != nil {
		m.q.Free()
	}

	if m.dir != nil {
		m.dir.Free()
	}

	if m.xOld != nil {
		m.xOld.Free()
	}

	if m.s != nil {
		m.s.Free()
	}

	if m.y != nil {
		m.y.Free()
	}

	for i := range m.sHist {

		if m.sHist[i] != nil {
			m.sHist[i].Free()
		}

		if m.yHist[i] != nil {
			m.yHist[i].Free()
		}
	}
}
