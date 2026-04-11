package engine

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("QrandGen", QrandGenNew, "Generic random Quantity. Args: seed, nComp (1 or 3), distType (0=normal, 1=uniform), updateMode (0=static, 1=per-step), mean, stddev.")
}

type randGen struct {
	seed      int64
	generator curand.Generator
	noise     *data.Slice
	step      int
}

type qrandQuantity struct {
	randGen
	nComp      int
	distType   int
	mean       float32
	stddev     float32
	updateMode int
}

func (q *qrandQuantity) update() {

	r := &q.randGen

	// --- init generator ---
	if r.generator == 0 {
		r.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		r.generator.SetSeed(r.seed)
	}

	// --- allocate buffer ---
	if r.noise == nil {
		r.noise = cuda.NewSlice(q.nComp, Mesh().Size())
		r.step = -1
	}

	if (q.updateMode == 0 && r.step >= 0) || (q.updateMode == 1 && r.step == NSteps) {
		return
	}

	// ========================================================
	// generation
	// ========================================================

	N := int64(Mesh().NCell())

	switch q.distType {

	// 0 = normal, 1 = uniform
	// does 3 curand calls. maybe faster to do 1 call and transform the data on the GPU?
	case 0:
		for c := 0; c < q.nComp; c++ {
			r.generator.GenerateNormal(uintptr(r.noise.DevPtr(c)), N, q.mean, q.stddev)
		}
	case 1:
		for c := 0; c < q.nComp; c++ {
			r.generator.GenerateUniform(uintptr(r.noise.DevPtr(c)), N)
		}
	default:
		panic(fmt.Sprintf("QrandGen: invalid distType %d", q.distType))
	}

	r.step = NSteps
}

func QrandGenNew(seed int, nComp int, distType int, updateMode int, mean, stddev float64) Quantity {

	return &qrandQuantity{
		randGen:    randGen{seed: int64(seed), step: -1},
		nComp:      nComp,
		distType:   distType,
		mean:       float32(mean),
		stddev:     float32(stddev),
		updateMode: updateMode,
	}
}

func (q *qrandQuantity) NComp() int   { return q.nComp }
func (q *qrandQuantity) Name() string { return "qrand" }
func (q *qrandQuantity) Unit() string { return "" }
func (q *qrandQuantity) EvalTo(dst *data.Slice) {
	q.update()
	data.Copy(dst, q.noise)
}
