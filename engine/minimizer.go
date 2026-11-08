package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	DmSamples int     = 10   // number of dm to keep for convergence check
	StopMaxDm float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
}

// fixed length FIFO. Items can be added but not removed
type fifoRing struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
	data  []float64
}

func FifoRing(length int) fifoRing {
	return fifoRing{data: make([]float64, length)}
}

func (r *fifoRing) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

func (r *fifoRing) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

type Minimizer struct {
	k        *data.Slice
	d        *data.Slice
	lastDm   fifoRing
	h        float32
	pc       *precond
	pcDeg    int
	pcScreen float64
}

func (mini *Minimizer) Step() {

	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
		if mini.pcDeg >= 2 && mini.pc == nil {
			mini.pc = newPrecond(m, mini.pcDeg, mini.pcScreen)
		}
		if mini.pc != nil {
			mini.d = cuda.Buffer(3, size)
			mini.pc.apply(mini.d, mini.k, m)
		} else {
			mini.d = mini.k
		}
	}

	k := mini.k
	d := mini.d
	h := mini.h
	if NSteps < 5 {
		util.Log(fmt.Sprintf("// step %d: h=%g  |k|2=%g  |d|2=%g  k.d=%g",
			NSteps, h, cuda.Dot(k, k), cuda.Dot(d, d), cuda.Dot(k, d)))
	}

	// save original magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// make descent along the (preconditioned) direction
	cuda.Minimize(m, m0, d, h)

	// calculate new torque for next step
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	var d0 *data.Slice
	if mini.pc != nil {
		d0 = cuda.Buffer(3, size)
		defer cuda.Recycle(d0)
		data.Copy(d0, d)
	}
	torqueFn(k)
	setMaxTorque(k) // report to user
	if mini.pc != nil {
		mini.pc.apply(d, k, m) // d = P k_new
	}

	// just to make the following readable
	dm := m0
	dk := k0

	// calculate step difference of m and k
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.) // reversed due to LLNoPrecess sign

	// get maxdiff and add to list
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user as LastErr

	// adjust next time step (Barzilai-Borwein, in the metric P^-1 if preconditioned)
	var nom, div float32
	switch {
	case mini.pc == nil && NSteps%2 == 0:
		nom = cuda.Dot(dm, dm)
		div = cuda.Dot(dm, dk)
	case mini.pc == nil:
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, dk)
	case NSteps%2 == 0: // BB1: s·P^-1 s / s·y, with P^-1 ≈ I + tau K
		mini.pc.applyInv(mini.pc.dB, dm, m)
		nom = cuda.Dot(dm, mini.pc.dB)
		div = cuda.Dot(dm, dk)
	default: // BB2: s·y / y·P y, with P y ≈ d_old - d_new
		cuda.Madd2(d0, d0, d, 1., -1.)
		nom = cuda.Dot(dm, dk)
		div = cuda.Dot(dk, d0)
	}
	if div != 0. {
		h := nom / div
		fellBack := false
		if h != h || h <= 0 { // NaN or non-positive curvature: grow the step
			h = 2 * mini.h
			fellBack = true
		}
		// cap the rotation angle per step
		if dmax := float32(cuda.MaxVecNorm(d)); dmax > 0 && h*dmax > 0.5 {
			h = 0.5 / dmax
		}
		mini.h = h
		if NSteps < 30 || NSteps%500 == 0 {
			util.Log(fmt.Sprintf("// BB step %d: nom=%g div=%g ratio=%g used h=%g fellBack=%v",
				NSteps, nom, div, nom/div, mini.h, fellBack))
		}
	} else {
		mini.h = 1e-4
	}

	M.normalize()

	// as a convention, time does not advance during relax
	NSteps++
}

func (mini *Minimizer) Free() {
	if mini.d != nil && mini.d != mini.k {
		cuda.Recycle(mini.d)
	}
	mini.d = nil
	if mini.k != nil {
		mini.k.Free()
		mini.k = nil
	}
	if mini.pc != nil {
		mini.pc.free()
		mini.pc = nil
	}
}

func Minimize() {
	Refer("exl2014")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the minimizer
	mini := Minimizer{
		h:        1e-4,
		k:        nil,
		lastDm:   FifoRing(DmSamples),
		pcDeg:    PrecondDegree,
		pcScreen: PrecondScreen}
	stepper = &mini
	util.Log(fmt.Sprintf("// precond active: %v (deg=%d, screen=%g)", mini.pc != nil, PrecondDegree, PrecondScreen))
	stepper = &mini
	stepper = &mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	}

	RunWhile(cond)
	pause = true
}
