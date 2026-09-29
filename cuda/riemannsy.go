package cuda

import "github.com/mumax/3/data"

// RiemannSY computes the Riemannian secant pair (s, y) in the tangent space at m0.
func RiemannSY(s, y, m0, m1, t0, t1 *data.Slice) {
	N := m1.Len()
	cfg := make1DConf(N)
	k_riemannsy_async(
		s.DevPtr(X), s.DevPtr(Y), s.DevPtr(Z),
		y.DevPtr(X), y.DevPtr(Y), y.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		t0.DevPtr(X), t0.DevPtr(Y), t0.DevPtr(Z),
		t1.DevPtr(X), t1.DevPtr(Y), t1.DevPtr(Z),
		N, cfg)
}
