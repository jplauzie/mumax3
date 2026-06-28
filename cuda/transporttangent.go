package cuda

import "github.com/mumax/3/data"

func TransportTangent(v, m0, m *data.Slice) {
	N := v.Len()
	cfg := make1DConf(N)

	k_transport_tangent_async(
		v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}
