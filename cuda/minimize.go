package cuda

import (
	"github.com/mumax/3/data"
)

// m = 1 / (4 + τ²(m x H)²) [{4 - τ²(m x H)²} m - 4τ(m x m x H)]
// note: torque from LLNoPrecess has negative sign
func Minimize(m, m0, torque *data.Slice, dt float32) {
	N := m.Len()
	cfg := make1DConf(N)

	k_minimize_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		dt, N, cfg)
}

func ParallelTransport(out, m0, m, k0 *data.Slice) {
	N := out.Len()
	cfg := make1DConf(N)

	k_parallel_transport_async(
		out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		k0.DevPtr(X), k0.DevPtr(Y), k0.DevPtr(Z),
		N, cfg)
}

func RiemannianDisplacement(out, m0, m *data.Slice) {
	N := out.Len()
	cfg := make1DConf(N)

	k_riemannian_displacement_async(
		out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}
