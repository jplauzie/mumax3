package cuda

import "github.com/mumax/3/data"

// MinimizeExp is Minimize with the exact rotation instead of the Cayley transform.
func MinimizeExp(m, m0, t *data.Slice, dt float32) {
	N := m.Len()
	cfg := make1DConf(N)
	k_minimizeexp_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		t.DevPtr(X), t.DevPtr(Y), t.DevPtr(Z),
		dt, N, cfg)
}
