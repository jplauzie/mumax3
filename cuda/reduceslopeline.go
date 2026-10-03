package cuda

import (
	"unsafe"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	slopePartials unsafe.Pointer // reducecfg.Grid.X floats
	slopeOut      unsafe.Pointer // one float32
	combineCfg    = &config{Grid: cu.Dim3{X: 1, Y: 1, Z: 1}, Block: cu.Dim3{X: 512, Y: 1, Z: 1}}
)

// SlopeAlongLine returns phi'(step) for phi(a) = E(normalize(wa + a*s)):
// sum over cells of (g . s) / |wa + step*s|, with g the gradient at the normalized trial point.
// Kahan-compensated per thread, warp-shuffle block reduction, final combine Kahan on the GPU.
func SlopeAlongLine(g, wa, s *data.Slice, ms MSlice, step float32) float64 {
	util.Argument(g.Size() == wa.Size() && g.Size() == s.Size())
	util.Argument(g.NComp() == 3 && wa.NComp() == 3 && s.NComp() == 3)

	nb := int(reducecfg.Grid.X)
	if slopePartials == nil {
		slopePartials = MemAlloc(int64(nb) * cu.SIZEOF_FLOAT32)
		slopeOut = MemAlloc(cu.SIZEOF_FLOAT32)
	}

	k_reduceslopeline_partial_async(
		g.DevPtr(X), g.DevPtr(Y), g.DevPtr(Z),
		wa.DevPtr(X), wa.DevPtr(Y), wa.DevPtr(Z),
		s.DevPtr(X), s.DevPtr(Y), s.DevPtr(Z),
		ms.DevPtr(0), ms.Mul(0),
		slopePartials, step, g.Len(), reducecfg)
	k_reduceslopeline_combine_async(slopeOut, slopePartials, nb, combineCfg)

	// mirror copyback(): async copy of 4 bytes, then synchronize the stream
	var out float32
	cu.MemcpyDtoH(unsafe.Pointer(&out), cu.DevicePtr(uintptr(slopeOut)), cu.SIZEOF_FLOAT32)
	return float64(out)
}
