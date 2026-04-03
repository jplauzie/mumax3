package engine

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("Qrand", QrandNew, "Quantity filled with standard normal random numbers (mean=0, stddev=1), fixed after first evaluation. Takes a seed and number of components (1 or 3).")
	DeclFunc("Qrandt", QrandtNew, "Quantity filled with standard normal random numbers (mean=0, stddev=1), refreshed every time step. Takes a seed and number of components (1 or 3).")
	DeclFunc("Qrandmul", QrandmulNew, "Multiply a Quantity pointwise by standard normal random numbers (mean=0, stddev=1), refreshed every time step. Takes a Quantity and a seed.")
	DeclFunc("Qrandnorm", QrandnormNew, "Quantity filled with uniform random numbers [0,1], refreshed every time step. Takes a seed and number of components (1 or 3).")
}

// ---- shared generator helper ----

type randGen struct {
	seed      int64
	generator curand.Generator
	noise     *data.Slice
}

func (r *randGen) ensureReady(nComp int) {
	if r.generator == 0 {
		r.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		r.generator.SetSeed(r.seed)
	}
	if r.noise == nil || r.noise.NComp() != nComp {
		if r.noise != nil {
			r.noise.Free()
		}
		r.noise = cuda.NewSlice(nComp, Mesh().Size())
	}
}

func (r *randGen) fillNormal(nComp int) {
	r.ensureReady(nComp)
	N := int64(Mesh().NCell())
	for c := 0; c < nComp; c++ {
		r.generator.GenerateNormal(uintptr(r.noise.DevPtr(c)), N, 0, 1)
	}
}

func (r *randGen) fillUniform(nComp int) {
	r.ensureReady(nComp)
	N := int64(Mesh().NCell())
	for c := 0; c < nComp; c++ {
		r.generator.GenerateUniform(uintptr(r.noise.DevPtr(c)), N)
	}
}

// ---- Qrand: fixed normal random field, generated once ----

type qrandQuantity struct {
	randGen
	nComp  int
	filled bool
	name   string
}

func QrandNew(seed int, nComp int) Quantity {
	return &qrandQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		name:    fmt.Sprintf("qrand_s%d_c%d", seed, nComp),
	}
}

func (q *qrandQuantity) NComp() int   { return q.nComp }
func (q *qrandQuantity) Name() string { return q.name }
func (q *qrandQuantity) Unit() string { return "" }

func (q *qrandQuantity) EvalTo(dst *data.Slice) {
	if !q.filled {
		q.fillNormal(q.nComp)
		q.filled = true
	}
	data.Copy(dst, q.noise)
}

// ---- Qrandt: normal random field refreshed every time step ----

type qrandtQuantity struct {
	randGen
	nComp int
	step  int
	name  string
}

func QrandtNew(seed int, nComp int) Quantity {
	return &qrandtQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		step:    -1,
		name:    fmt.Sprintf("qrandt_s%d_c%d", seed, nComp),
	}
}

func (q *qrandtQuantity) NComp() int   { return q.nComp }
func (q *qrandtQuantity) Name() string { return q.name }
func (q *qrandtQuantity) Unit() string { return "" }

func (q *qrandtQuantity) EvalTo(dst *data.Slice) {
	if NSteps != q.step {
		q.fillNormal(q.nComp)
		q.step = NSteps
	}
	data.Copy(dst, q.noise)
}

// ---- Qrandmul: pointwise multiply a Quantity by per-step normal random values ----

type qrandmulQuantity struct {
	randGen
	src  Quantity
	step int
	name string
}

func QrandmulNew(src Quantity, seed int) Quantity {
	return &qrandmulQuantity{
		randGen: randGen{seed: int64(seed)},
		src:     src,
		step:    -1,
		name:    fmt.Sprintf("qrandmul_s%d", seed),
	}
}

func (q *qrandmulQuantity) NComp() int   { return q.src.NComp() }
func (q *qrandmulQuantity) Name() string { return q.name }
func (q *qrandmulQuantity) Unit() string { return "" }

func (q *qrandmulQuantity) EvalTo(dst *data.Slice) {
	if NSteps != q.step {
		q.fillNormal(q.src.NComp())
		q.step = NSteps
	}
	src := ValueOf(q.src)
	defer cuda.Recycle(src)
	cuda.Mul(dst, src, q.noise)
}

// ---- Qrandnorm: uniform [0,1] random field refreshed every time step ----

type qrandnormQuantity struct {
	randGen
	nComp int
	step  int
	name  string
}

func QrandnormNew(seed int, nComp int) Quantity {
	return &qrandnormQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		step:    -1,
		name:    fmt.Sprintf("qrandnorm_s%d_c%d", seed, nComp),
	}
}

func (q *qrandnormQuantity) NComp() int   { return q.nComp }
func (q *qrandnormQuantity) Name() string { return q.name }
func (q *qrandnormQuantity) Unit() string { return "" }

func (q *qrandnormQuantity) EvalTo(dst *data.Slice) {
	if NSteps != q.step {
		q.fillUniform(q.nComp)
		q.step = NSteps
	}
	data.Copy(dst, q.noise)
}
