package cuda

import (
	"github.com/mumax/3/data"
)

// h = Hess[d] ~= -P_m[(t1 - t0) * invEps], see trhessvec.cu.
func TRHessVec(h, t1, t0, m *data.Slice, invEps float32) {
	N := h.Len()
	cfg := make1DConf(N)

	k_trhessvec_async(h.DevPtr(X), h.DevPtr(Y), h.DevPtr(Z),
		t1.DevPtr(X), t1.DevPtr(Y), t1.DevPtr(Z),
		t0.DevPtr(X), t0.DevPtr(Y), t0.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		invEps, N, cfg)
}

// v = velocity at the end of the Cayley retraction path of c*s starting at m0, see trendvel.cu.
func TREndVel(v, m0, s *data.Slice, c float32) {
	N := v.Len()
	cfg := make1DConf(N)

	k_trendvel_async(v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		s.DevPtr(X), s.DevPtr(Y), s.DevPtr(Z),
		c, N, cfg)
}
