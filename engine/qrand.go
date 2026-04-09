package engine

import (
	"fmt"
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Qrand", QrandNew, "Quantity filled with normal random numbers, fixed after first evaluation. Takes seed, number of components (1 or 3), mean, and stddev.")
	DeclFunc("Qrandt", QrandtNew, "Quantity filled with normal random numbers, refreshed every time step. Takes seed, number of components (1 or 3), mean, and stddev.")
	DeclFunc("Qrandut", QrandutNew, "Quantity filled with uniform random numbers [0,1], refreshed every time step. Takes seed and number of components (1 or 3).")
	DeclFunc("Qranduni", QranduniNew, "Quantity filled with uniform random numbers [0,1], fixed after first evaluation. Takes seed and number of components (1 or 3).")
	DeclFunc("Qstat", QstatNew, "Compute and print mean and stddev of a Quantity every N steps. Takes a Quantity and interval.")
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

func (r *randGen) fillNormal(nComp int, mean, stddev float32) {
	r.ensureReady(nComp)
	N := int64(Mesh().NCell())
	for c := 0; c < nComp; c++ {
		r.generator.GenerateNormal(uintptr(r.noise.DevPtr(c)), N, mean, stddev)
	}
}

func (r *randGen) fillUniform(nComp int) {
	r.ensureReady(nComp)
	N := int64(Mesh().NCell())
	for c := 0; c < nComp; c++ {
		r.generator.GenerateUniform(uintptr(r.noise.DevPtr(c)), N)
	}
}

// ---- Qrand: fixed normal random field ----

type qrandQuantity struct {
	randGen
	nComp  int
	mean   float32
	stddev float32
	filled bool
	name   string
}

func QrandNew(seed int, nComp int, mean, stddev float64) Quantity {
	return &qrandQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		mean:    float32(mean),
		stddev:  float32(stddev),
		name:    fmt.Sprintf("qrand_s%d_c%d_m%.2f_s%.2f", seed, nComp, mean, stddev),
	}
}

func (q *qrandQuantity) NComp() int   { return q.nComp }
func (q *qrandQuantity) Name() string { return q.name }
func (q *qrandQuantity) Unit() string { return "" }

func (q *qrandQuantity) EvalTo(dst *data.Slice) {
	if !q.filled {
		q.fillNormal(q.nComp, q.mean, q.stddev)
		q.filled = true
	}
	data.Copy(dst, q.noise)
}

// ---- Qrandt: time-varying normal random field ----

type qrandtQuantity struct {
	randGen
	nComp  int
	mean   float32
	stddev float32
	step   int
	name   string
}

func QrandtNew(seed int, nComp int, mean, stddev float64) Quantity {
	return &qrandtQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		mean:    float32(mean),
		stddev:  float32(stddev),
		step:    -1,
		name:    fmt.Sprintf("qrandt_s%d_c%d_m%.2f_s%.2f", seed, nComp, mean, stddev),
	}
}

func (q *qrandtQuantity) NComp() int   { return q.nComp }
func (q *qrandtQuantity) Name() string { return q.name }
func (q *qrandtQuantity) Unit() string { return "" }

func (q *qrandtQuantity) EvalTo(dst *data.Slice) {
	if NSteps != q.step {
		q.fillNormal(q.nComp, q.mean, q.stddev)
		q.step = NSteps
	}
	data.Copy(dst, q.noise)
}

// ---- Qrandut: time-varying uniform [0,1] random field ----

type qrandutQuantity struct {
	randGen
	nComp int
	step  int
	name  string
}

func QrandutNew(seed int, nComp int) Quantity {
	return &qrandutQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		step:    -1,
		name:    fmt.Sprintf("qrandut_s%d_c%d", seed, nComp),
	}
}

func (q *qrandutQuantity) NComp() int   { return q.nComp }
func (q *qrandutQuantity) Name() string { return q.name }
func (q *qrandutQuantity) Unit() string { return "" }

func (q *qrandutQuantity) EvalTo(dst *data.Slice) {
	if NSteps != q.step {
		q.fillUniform(q.nComp)
		q.step = NSteps
	}
	data.Copy(dst, q.noise)
}

// ---- Qranduni: fixed uniform [0,1] random field ----

type qranduniQuantity struct {
	randGen
	nComp  int
	filled bool
	name   string
}

func QranduniNew(seed int, nComp int) Quantity {
	return &qranduniQuantity{
		randGen: randGen{seed: int64(seed)},
		nComp:   nComp,
		name:    fmt.Sprintf("qranduni_s%d_c%d", seed, nComp),
	}
}

func (q *qranduniQuantity) NComp() int   { return q.nComp }
func (q *qranduniQuantity) Name() string { return q.name }
func (q *qranduniQuantity) Unit() string { return "" }

func (q *qranduniQuantity) EvalTo(dst *data.Slice) {
	if !q.filled {
		q.fillUniform(q.nComp)
		q.filled = true
	}
	data.Copy(dst, q.noise)
}

// ---- Qstat: runtime statistics + histogram ----

type qstat struct {
	q        Quantity
	interval int
	lastStep int
	buf      *data.Slice
	name     string
	bins     int
}

func QstatNew(q Quantity, interval int) Quantity {
	return &qstat{
		q:        q,
		interval: interval,
		lastStep: -1,
		name:     fmt.Sprintf("qstat_%T", q),
		bins:     50, // default histogram bins
	}
}

func (s *qstat) NComp() int   { return s.q.NComp() }
func (s *qstat) Name() string { return s.name }
func (s *qstat) Unit() string { return "" }

func (s *qstat) EvalTo(dst *data.Slice) {
	// ensure buffer
	if s.buf == nil || s.buf.NComp() != s.q.NComp() {
		if s.buf != nil {
			s.buf.Free()
		}
		s.buf = data.NewSlice(s.q.NComp(), Mesh().Size())
	}

	// evaluate underlying quantity
	s.q.EvalTo(s.buf)

	// periodic stats
	if s.interval > 0 && NSteps%s.interval == 0 && NSteps != s.lastStep {
		s.computeStats()
		s.lastStep = NSteps
	}

	// pass-through
	data.Copy(dst, s.buf)
}

func (s *qstat) computeStats() {
	mesh := Mesh()
	nx, ny, nz := mesh.Size()[0], mesh.Size()[1], mesh.Size()[2]
	nComp := s.buf.NComp()
	nCell := float64(nx * ny * nz)

	means := make([]float64, nComp)
	stds := make([]float64, nComp)

	// compute per-component stats
	for c := 0; c < nComp; c++ {
		compData := s.buf.Comp(c).HostCopy().Scalars() // 3D: [z][y][x]

		// compute mean
		var sum float64
		for iz := 0; iz < nz; iz++ {
			for iy := 0; iy < ny; iy++ {
				for ix := 0; ix < nx; ix++ {
					sum += float64(compData[iz][iy][ix])
				}
			}
		}
		mean := sum / nCell
		means[c] = mean

		// compute std
		var varsum float64
		for iz := 0; iz < nz; iz++ {
			for iy := 0; iy < ny; iy++ {
				for ix := 0; ix < nx; ix++ {
					diff := float64(compData[iz][iy][ix]) - mean
					varsum += diff * diff
				}
			}
		}
		stds[c] = math.Sqrt(varsum / nCell)

		// print histogram
		s.printHistogram(c, compData, mean, stds[c])
	}

	// print summary
	util.Log("Qstat", s.name, "step", NSteps, "mean", means, "std", stds)
}

func (s *qstat) printHistogram(comp int, data [][][]float32, mean, std float64) {
	nx, ny, nz := Mesh().Size()[0], Mesh().Size()[1], Mesh().Size()[2]

	// histogram range
	min := mean - 4*std
	max := mean + 4*std
	if std == 0 {
		min = mean - 1
		max = mean + 1
	}

	bins := s.bins
	counts := make([]int, bins)

	// fill histogram
	for iz := 0; iz < nz; iz++ {
		for iy := 0; iy < ny; iy++ {
			for ix := 0; ix < nx; ix++ {
				v := float64(data[iz][iy][ix])
				b := int((v - min) / (max - min) * float64(bins))
				if b >= 0 && b < bins {
					counts[b]++
				}
			}
		}
	}

	// print histogram
	util.Log("Qstat", s.name, "step", NSteps, "histogram", fmt.Sprintf("component %d", comp))
	for i := 0; i < bins; i++ {
		x0 := min + (max-min)*float64(i)/float64(bins)
		x1 := min + (max-min)*float64(i+1)/float64(bins)
		util.Log("histogram", "comp", comp, fmt.Sprintf("[% .3e, % .3e)", x0, x1), counts[i])
	}
}
