package cuda

import "github.com/mumax/3/data"

// ProjectTangent projects v onto the tangent plane of m.
func ProjectTangent(v, m *data.Slice) {
	N := v.Len()
	cfg := make1DConf(N)

	k_project_tangent_async(
		v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		N, cfg)
}
