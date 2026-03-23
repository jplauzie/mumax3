package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("CustomQuantity", CustomQuantity, "Creates a custom Quantity from the user provided (scalar or vector) slice.")
	DeclFunc("probe", Probe, "Print value of a quantity at cell (ix, iy, iz)")
}

func CustomQuantity(inSlice *data.Slice) Quantity {
	util.Assert(inSlice.NComp() == 1 || inSlice.NComp() == 3)
	size := Mesh().Size()
	sliceSize := inSlice.Size()
	util.Assert(size[X] == sliceSize[X] && size[Y] == sliceSize[Y] && size[Z] == sliceSize[Z])

	retQuant := &customQuantity{nil, size}
	if inSlice.NComp() == 1 {
		retQuant.customquant = cuda.NewSlice(SCALAR, size)
	} else {
		retQuant.customquant = cuda.NewSlice(VECTOR, size)
	}

	data.Copy(retQuant.customquant, inSlice)
	return retQuant
}

type customQuantity struct {
	customquant *data.Slice
	size        [3]int
}

func (q *customQuantity) NComp() int {
	return q.customquant.NComp()
}

func (q *customQuantity) EvalTo(dst *data.Slice) {
	util.Assert(dst.NComp() == q.customquant.NComp())
	data.Copy(dst, q.customquant)
}

func Probe(q Quantity, ix, iy, iz int) {
	size := SizeOf(q)
	nx, ny, nz := size[0], size[1], size[2]

	// Bounds check
	if ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz {
		util.Log("probe:", NameOf(q), "index out of bounds:",
			ix, iy, iz, "size:", nx, ny, nz)
		return
	}

	// Evaluate on GPU
	buf := ValueOf(q)
	defer cuda.Recycle(buf)

	// Copy to CPU
	hostBuf := buf.HostCopy()
	defer hostBuf.Free()

	ncomp := q.NComp()

	if ncomp == 1 {
		val := hostBuf.Get(0, ix, iy, iz)
		util.Log(NameOf(q),
			"[", ix, iy, iz, "] =",
			val, UnitOf(q))
	} else {
		values := make([]float64, ncomp)
		for c := 0; c < ncomp; c++ {
			values[c] = hostBuf.Get(c, ix, iy, iz)
		}

		util.Log(NameOf(q),
			"[", ix, iy, iz, "] =",
			values, UnitOf(q))
	}
}
