package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	PrecondDegree int     = 0 // Chebyshev degree; 0 or 1 disables
	PrecondScreen float64 = 3 // screening length of the preconditioner, in cells
)

func init() {
	DeclVar("MinimizerPrecondDegree", &PrecondDegree, "Degree of the exchange (Sobolev) preconditioner polynomial for Minimize. Costs degree-1 stencil passes per step. 0 disables (default).")
	DeclVar("MinimizerPrecondScreen", &PrecondScreen, "Screening length (in cells) of the Minimize preconditioner (default 3).")
}

// Approximates P = (I + tau*K)^-1 with a fixed-degree Chebyshev polynomial.
type precond struct {
	deg          int
	tau          float32
	theta, delta float64
	res, dA, dB  *data.Slice
}

func newPrecond(m *data.Slice, deg int, screen float64) *precond {
	size := m.Size()
	ndim := 0
	for _, n := range size {
		if n > 1 {
			ndim++
		}
	}
	if deg < 2 || ndim == 0 || screen <= 0 {
		return nil
	}
	buf := cuda.Buffer(1, size)
	defer cuda.Recycle(buf)
	cuda.PrecondStiffness(buf, m, lex2.Gpu(), regions.Gpu(), Mesh())
	dmax := float64(cuda.MaxAbs(buf))
	if dmax == 0 { // no exchange coupling anywhere: nothing to precondition
		return nil
	}
	s2 := screen * screen
	nd := float64(ndim)
	b := 1 + 4*nd*s2 // Gershgorin upper bound of spec(I + tau K); lower bound is 1
	return &precond{
		deg:   deg,
		tau:   float32(s2 * 2 * nd / dmax),
		theta: (b + 1) / 2,
		delta: (b - 1) / 2,
		res:   cuda.Buffer(3, size),
		dA:    cuda.Buffer(3, size),
		dB:    cuda.Buffer(3, size),
	}
}

func (p *precond) free() {
	cuda.Recycle(p.res)
	cuda.Recycle(p.dA)
	cuda.Recycle(p.dB)
}

// z = P r, projected tangent to m, frozen spins zeroed. z must not alias r.
func (p *precond) apply(z, r, m *data.Slice) {
	// in (p *precond) apply, first line

	aex, reg, mesh := lex2.Gpu(), regions.Gpu(), Mesh()
	sigma := p.theta / p.delta
	rho := 1 / sigma

	cuda.Madd2(z, r, r, float32(1/p.theta), 0) // z = d0 = r/theta
	data.Copy(p.dA, z)
	data.Copy(p.res, r)

	d, dn := p.dA, p.dB
	for k := 1; k < p.deg; k++ {
		rhoNew := 1 / (2*sigma - rho)
		c1 := float32(rhoNew * rho)
		c2 := float32(2 * rhoNew / p.delta)
		cuda.PrecondStep(dn, d, p.res, z, m, aex, reg, p.tau, c1, c2, mesh)
		d, dn = dn, d
		rho = rhoNew
	}
	cuda.ProjectTangent(z, m)
	FreezeSpins(z)
}

// scratch = (I + tau K) v ≈ P^-1 v. scratch must not alias v.
func (p *precond) applyInv(scratch, v, m *data.Slice) {
	cuda.PrecondApply(scratch, v, m, lex2.Gpu(), regions.Gpu(), p.tau, Mesh())
}
