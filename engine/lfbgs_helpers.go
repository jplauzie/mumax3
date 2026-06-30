package engine

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func (m *LBFGSMinimizer) EnergyAndGradient(g *data.Slice) float64 {

	// With Precess=false, SetTorque returns
	// -(m × (m × Heff)), i.e. the projected energy gradient
	// up to a sign convention.
	torqueFn(g)

	return GetTotalEnergy()
}

func dot(a, b *data.Slice) float32 {
	return cuda.Dot(a, b)
}

func norm(a *data.Slice) float32 {
	v := cuda.Dot(a, a)
	return float32(math.Sqrt(float64(v)))
}

func maxNorm(a *data.Slice) float32 {
	return cuda.MaxVecNorm(a)
}

func copySlice(dst, src *data.Slice) {
	data.Copy(dst, src)
}

func gradientDifference(dst, g, gold *data.Slice) {
	cuda.Madd2(dst, g, gold, 1, -1)
}

func magnetizationDifference(dst, m, mold *data.Slice) {
	cuda.Madd2(dst, m, mold, 1, -1)
}

func (m *LBFGSMinimizer) resetHistory() {

	m.iter = 0
	m.H0 = 1

	for i := range m.alpha {
		m.alpha[i] = 0
		m.rho[i] = 0
	}
}

ys = dot(y,s)

if (...) skip

else {

    H0 = ys/dot(y,y)

    push s,y

}

func (m *LBFGSMinimizer) updateHistory() {

	ys := dot(m.y, m.s)

	if ys <= 0 {
		return
	}

	yy := dot(m.y, m.y)

	if yy == 0 {
		return
	}

	m.H0 = ys / yy

	if m.iter < lbfgsHistory {

		copySlice(m.sHist[m.iter], m.s)
		copySlice(m.yHist[m.iter], m.y)

		m.rho[m.iter] = 1 / ys

		m.iter++

		return
	}

	// shift oldest history out

	for i := 0; i < lbfgsHistory-1; i++ {

		copySlice(m.sHist[i], m.sHist[i+1])
		copySlice(m.yHist[i], m.yHist[i+1])

		m.rho[i] = m.rho[i+1]
	}

	copySlice(m.sHist[lbfgsHistory-1], m.s)
	copySlice(m.yHist[lbfgsHistory-1], m.y)

	m.rho[lbfgsHistory-1] = 1 / ys
}

func (m *LBFGSMinimizer) ComputeSearchDirection() {

	copySlice(m.q, m.g)

	k := m.iter
	if k > lbfgsHistory {
		k = lbfgsHistory
	}

	//------------------------------------------
	// first loop
	//------------------------------------------

	for i := k - 1; i >= 0; i-- {

		m.alpha[i] = m.rho[i] * dot(m.sHist[i], m.q)

		// q -= alpha*y

		cuda.Axpby(
			m.q,
			m.q,
			m.yHist[i],
			1,
			-m.alpha[i],
		)
	}

	//------------------------------------------
	// initial Hessian scaling
	//------------------------------------------

	cuda.Scale(m.q, m.H0)

	//------------------------------------------
	// second loop
	//------------------------------------------

	for i := 0; i < k; i++ {

		beta := m.rho[i] * dot(m.yHist[i], m.q)

		// q += (alpha-beta)s

		cuda.Axpby(
			m.q,
			m.q,
			m.sHist[i],
			1,
			m.alpha[i]-beta,
		)
	}
}

func (m *LBFGSMinimizer) EnsureDescent() {

	phiPrime := -dot(m.g, m.q)

	if phiPrime > 0 {

		copySlice(m.q, m.g)

		m.resetHistory()
	}
}

func (m *LBFGSMinimizer) SaveCurrentState() {

	copySlice(m.xOld, M.Buffer())

	copySlice(m.gOld, m.g)
}

func (m *LBFGSMinimizer) BuildHistoryVectors() {

	magnetizationDifference(
		m.s,
		M.Buffer(),
		m.xOld,
	)

	gradientDifference(
		m.y,
		m.g,
		m.gOld,
	)
}

func (m *LBFGSMinimizer) ComputeSearchDirection() {

	copySlice(m.q, m.g)

	k := m.iter
	if k > lbfgsHistory {
		k = lbfgsHistory
	}

	// First loop
	for i := k - 1; i >= 0; i-- {

		m.alpha[i] = m.rho[i] * dot(m.sHist[i], m.q)

		// q -= alpha_i * y_i
		cuda.Madd2(
			m.q,
			m.q,
			m.yHist[i],
			1,
			-m.alpha[i],
		)
	}

	// q = H0 * q
	cuda.Scale(m.q, m.H0)

	// Second loop
	for i := 0; i < k; i++ {

		beta := m.rho[i] * dot(m.yHist[i], m.q)

		// q += (alpha-beta) s_i
		cuda.Madd2(
			m.q,
			m.q,
			m.sHist[i],
			1,
			m.alpha[i]-beta,
		)
	}
}