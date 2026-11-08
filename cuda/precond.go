package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

func exchW(mesh *data.Mesh) (wx, wy, wz float32) {
	c := mesh.CellSize()
	return float32(2 / (c[X] * c[X])), float32(2 / (c[Y] * c[Y])), float32(2 / (c[Z] * c[Z]))
}

// dst (1 component) = diagonal of the exchange stencil K, 0 in empty cells.
func PrecondStiffness(dst, m *data.Slice, Aex SymmLUT, regions *Bytes, mesh *data.Mesh) {
	wx, wy, wz := exchW(mesh)
	N := mesh.Size()
	cfg := make3DConf(N)
	k_precondstiff_async(dst.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// dst = v + tau*K v. dst must not alias v.
func PrecondApply(dst, v, m *data.Slice, Aex SymmLUT, regions *Bytes, tau float32, mesh *data.Mesh) {
	wx, wy, wz := exchW(mesh)
	N := mesh.Size()
	cfg := make3DConf(N)
	k_precondapply_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex), regions.Ptr,
		tau, wx, wy, wz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// One fused Chebyshev step: res -= A dOld; dNew = c1*dOld + c2*res; z += dNew.
func PrecondStep(dNew, dOld, res, z, m *data.Slice, Aex SymmLUT, regions *Bytes,
	tau, c1, c2 float32, mesh *data.Mesh) {
	wx, wy, wz := exchW(mesh)
	N := mesh.Size()
	cfg := make3DConf(N)
	k_precondstep_async(dNew.DevPtr(X), dNew.DevPtr(Y), dNew.DevPtr(Z),
		dOld.DevPtr(X), dOld.DevPtr(Y), dOld.DevPtr(Z),
		res.DevPtr(X), res.DevPtr(Y), res.DevPtr(Z),
		z.DevPtr(X), z.DevPtr(Y), z.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(Aex), regions.Ptr,
		tau, c1, c2, wx, wy, wz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

// v -= (v·m) m, zero where m == 0.
func ProjectTangent(v, m *data.Slice) {
	N := v.Len()
	cfg := make1DConf(N)
	k_projecttangent_async(v.DevPtr(X), v.DevPtr(Y), v.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z), N, cfg)
}
