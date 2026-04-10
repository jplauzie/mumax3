package engine

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("QrandGen", QrandGenNew, "Generic random Quantity. Args: seed, nComp (1 or 3), distType, updateMode, mean, stddev. distType: 0=normal, 1=uniform. updateMode: 0=static, 1=per-step.")
	r.step = -1
}

// ============================================================
// 🔧 Constants (keep scripts readable)
// ============================================================

const (
	DistNormal  = 0
	DistUniform = 1
)

const (
	UpdateStatic = 0
	UpdateStep   = 1
)

// ============================================================
// 🔧 Shared random generator
// ============================================================

type randGen struct {
	seed      int64
	generator curand.Generator
	noise     *data.Slice
	step      int
}

func (r *randGen) ensureReady(nComp int) {
	if r.generator == 0 {
		r.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		r.generator.SetSeed(r.seed)
	}
	if r.noise == nil {
		r.noise = cuda.NewSlice(nComp, Mesh().Size())
		r.step = -1
	}
}

// ============================================================
// 🔁 Policies
// ============================================================

type refreshPolicy func(lastStep, currentStep int) bool

func refreshEveryStep(last, current int) bool {
	return last != current
}

func refreshOnce(last, current int) bool {
	return last < 0
}

// ============================================================
// 🎲 Generic random quantity
// ============================================================

type qrandQuantity struct {
	randGen
	nComp  int
	name   string
	policy refreshPolicy
	fill   func()
}

func (q *qrandQuantity) NComp() int   { return q.nComp }
func (q *qrandQuantity) Name() string { return q.name }
func (q *qrandQuantity) Unit() string { return "" }

func (q *qrandQuantity) EvalTo(dst *data.Slice) {
	q.update(NSteps, q.policy, q.fill)
	data.Copy(dst, q.noise)
}

// unified update
func (r *randGen) update(step int, policy refreshPolicy, refill func()) {

	if r.noise == nil {
		refill()
		r.step = step
		return
	}

	if !policy(r.step, step) {
		return
	}

	refill()
	r.step = step
}

// ============================================================
// 🧪 Fill helpers
// ============================================================

func makeNormalFiller(r *randGen, nComp int, mean, stddev float32) func() {
	return func() {
		r.ensureReady(nComp)
		N := int64(Mesh().NCell())
		for c := 0; c < nComp; c++ {
			r.generator.GenerateNormal(uintptr(r.noise.DevPtr(c)), N, mean, stddev)
		}
	}
}

func makeUniformFiller(r *randGen, nComp int) func() {
	return func() {
		r.ensureReady(nComp)
		N := int64(Mesh().NCell())
		for c := 0; c < nComp; c++ {
			r.generator.GenerateUniform(uintptr(r.noise.DevPtr(c)), N)
		}
	}
}

// ============================================================
// 🎲 Single constructor
// ============================================================

func QrandGenNew(seed int, nComp int, distType int, updateMode int, mean, stddev float64) Quantity {

	q := &qrandQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
	}

	// -------------------------
	// Distribution selection
	// -------------------------

	switch distType {
	case DistNormal:
		q.fill = makeNormalFiller(&q.randGen, nComp, float32(mean), float32(stddev))
	case DistUniform:
		q.fill = makeUniformFiller(&q.randGen, nComp)
	default:
		panic(fmt.Sprintf("QrandGen: unknown distType %d", distType))
	}

	// -------------------------
	// Update policy
	// -------------------------

	switch updateMode {
	case UpdateStatic:
		q.policy = refreshOnce
	case UpdateStep:
		q.policy = refreshEveryStep
	default:
		panic(fmt.Sprintf("QrandGen: unknown updateMode %d", updateMode))
	}

	// -------------------------
	// Naming (important for debugging)
	// -------------------------

	distStr := "unknown"
	if distType == DistNormal {
		distStr = "normal"
	} else if distType == DistUniform {
		distStr = "uniform"
	}

	updateStr := "unknown"
	if updateMode == UpdateStatic {
		updateStr = "static"
	} else if updateMode == UpdateStep {
		updateStr = "step"
	}

	if distType == DistNormal {
		q.name = fmt.Sprintf("qrand_%s_%s_s%d_c%d_m%.2f_s%.2f",
			distStr, updateStr, seed, nComp, mean, stddev)
	} else {
		q.name = fmt.Sprintf("qrand_%s_%s_s%d_c%d",
			distStr, updateStr, seed, nComp)
	}

	return q
}
