package engine

// Minimize follows the steepest descent method as per Exl et al., JAP 115, 17D118 (2014).

import (
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	DmSamples             int     = 10   // number of dm to keep for convergence check
	StopMaxDm             float64 = 1e-6 // stop minimizer if sampled dm is smaller than this
	MinimizeWallClockTime float64 = -1.0 // wall-clock time limit for minimization
	MinimizeConverged     bool           // true if minimize converged, and false if the maximum wall-clock time is reached

	// experimental options
	MinimizerTwoStep bool // use two-step (Ford-Moghrabi style) secant instead of plain BB
	MinimizerRiemann int  // 0 = off, 1 = Riemannian s and y, 2 = Riemannian s only, 3 = Riemannian y only
	MinimizerExact   bool // use exact rotation instead of the Cayley transform

	// Zhang-Xu energy-corrected BB
	MinimizerZX      int         // 0 off, 1 BB1+BB2, 2 BB1 only (even steps), 3 BB2 only (odd steps)
	MinimizerZXScale float64 = 1 // energy scale fix: set to 1/c if the torque carries a prefactor c
	MinimizerZXClip  float64 = 1 // use plain BB when |theta| > clip*|s.y|; <=0 disables the gate
	MinimizerZXDebug bool        // print calibration diagnostics

	// Initial step size by strong-Wolfe (More-Thuente) line search
	MinimizerLS         bool          // use a line search for the initial step size of each Minimize call
	MinimizerLSAngle    float64 = 0.1 // starting trial: max rotation angle (rad) = atan(stp*max|t|)
	MinimizerLSMaxAngle float64 = 45  // degrees; cap on the rotation angle of any trial step (< 90)
	MinimizerLSDebug    bool

	// Per-step cap on |h|*max|t| (rad). 0 = off.
	MinimizerMaxAngle float64

	// diagnostics (reset at the start of each Minimize call)
	MinimizerFallbacks   int  // steps where two-step failed and plain BB was used (or NaN/Inf guard fired)
	MinimizerNegSteps    int  // steps where h < 0
	MinimizerZXUsed      int  // steps where theta was applied
	MinimizerZXGated     int  // steps where the gate reverted to plain BB
	MinimizerZXSignFlips int  // steps where sign(s.y_hat) != sign(s.y)
	MinimizerAngleClamps int  // steps where the angle cap was applied
	MinimizerNaNGuard    bool // replace a NaN/Inf step size with 1e-4 instead of letting it propagate (experimental)
)

func init() {
	DeclFunc("Minimize", Minimize, "Use steepest conjugate gradient method to minimize the total energy. Returns true if convergence is reached, or false if the wall-clock time limit is exceeded. The wall-clock time limit is disabled by default.")
	DeclVar("MinimizerStop", &StopMaxDm, "Stopping max dM for Minimize")
	DeclVar("MinimizerSamples", &DmSamples, "Number of max dM to collect for Minimize convergence check.")
	DeclVar("MinimizeWallClockTime", &MinimizeWallClockTime, "Wall-clock time limit (seconds) for Minimize that will interrupt the minimization if exceeded. Set to -1 (default) to disable. An interrupted minimization does not guarantee a correct solution.")
	DeclVar("MinimizerTwoStep", &MinimizerTwoStep, "Use two-step secant for the Minimize step size (experimental)")
	DeclVar("MinimizerRiemann", &MinimizerRiemann, "Riemannian secant pair for Minimize: 0 off, 1 s and y, 2 s only, 3 y only (experimental)")
	DeclVar("MinimizerExact", &MinimizerExact, "Use exact rotation instead of Cayley transform in Minimize (experimental)")
	DeclVar("MinimizerZX", &MinimizerZX, "Zhang-Xu energy-corrected BB: 0 off, 1 both, 2 BB1 only, 3 BB2 only (experimental)")
	DeclVar("MinimizerZXScale", &MinimizerZXScale, "Energy scaling for Zhang-Xu (set to 1/c if torque has prefactor c)")
	DeclVar("MinimizerZXClip", &MinimizerZXClip, "Zhang-Xu gate: plain BB when |theta| > clip*|s.y|; <=0 disables")
	DeclVar("MinimizerZXDebug", &MinimizerZXDebug, "Print Zhang-Xu calibration diagnostics")
	DeclVar("MinimizerLS", &MinimizerLS, "Strong-Wolfe line search for the initial Minimize step size (experimental)")
	DeclVar("MinimizerLSAngle", &MinimizerLSAngle, "Line search: starting trial max rotation angle (rad)")
	DeclVar("MinimizerLSMaxAngle", &MinimizerLSMaxAngle, "Line search: max rotation angle (degrees, < 90) of any trial step")
	DeclVar("MinimizerLSDebug", &MinimizerLSDebug, "Print line search diagnostics")
	DeclVar("MinimizerMaxAngle", &MinimizerMaxAngle, "Cap on |h|*max|torque| (rad) for every Minimize step, 0 disables (experimental)")
	DeclVar("MinimizerFallbacks", &MinimizerFallbacks, "Number of two-step fallbacks to plain BB (diagnostic)")
	DeclVar("MinimizerNegSteps", &MinimizerNegSteps, "Number of negative step sizes (diagnostic)")
	DeclVar("MinimizerZXUsed", &MinimizerZXUsed, "Zhang-Xu steps applied (diagnostic)")
	DeclVar("MinimizerZXGated", &MinimizerZXGated, "Zhang-Xu steps gated to plain BB (diagnostic)")
	DeclVar("MinimizerZXSignFlips", &MinimizerZXSignFlips, "Steps where Zhang-Xu changed the sign of s.y (diagnostic)")
	DeclVar("MinimizerAngleClamps", &MinimizerAngleClamps, "Steps where the angle cap was applied (diagnostic)")
	DeclVar("MinimizerNaNGuard", &MinimizerNaNGuard, "Replace NaN/Inf step sizes with 1e-4 in Minimize (experimental)")
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
	k      *data.Slice // torque saved to calculate time step
	prevDm *data.Slice // previous secant s (two-step only)
	prevDk *data.Slice // previous secant y (two-step only)
	prevA  float32     // |s_{k-1}| (two-step only)
	prevE  float64     // energy (units of Ms*V) at the previous m (Zhang-Xu only)
	lastDm fifoRing
	h      float32
}

// BB step from a pair (s, y); alternates BB1/BB2 with NSteps parity.
// The sign is preserved: negative curvature gives a negative step.
// ok is false only if the denominator is exactly zero.
func bbStep(s, y *data.Slice) (h float32, ok bool) {
	var nom, div float32
	if NSteps%2 == 0 {
		nom = cuda.Dot(s, s)
		div = cuda.Dot(s, y)
	} else {
		nom = cuda.Dot(s, y)
		div = cuda.Dot(y, y)
	}
	if div != 0 {
		return nom / div, true
	}
	return 0, false
}

// Energy in units of Ms*V (times MinimizerZXScale), so that the gradient is -t.
// Assumes uniform Msat and cell volume.
// Check the names Msat.Average() and Mesh().CellSize() against your fork.
func zxEnergy() float64 {
	c := Mesh().CellSize()
	kappa := Msat.Average() * c[0] * c[1] * c[2] * MinimizerZXScale
	return GetTotalEnergy() / kappa
}

// Zhang-Xu corrected BB step. dE = Ebar_k - Ebar_{k+1}.
func bbStepZX(s, y, tNew *data.Slice, dE float64) (float32, bool) {
	ss := float64(cuda.Dot(s, s))
	sy := float64(cuda.Dot(s, y))
	yy := float64(cuda.Dot(y, y))
	tns := float64(cuda.Dot(tNew, s))

	theta := 6*dE - 3*(sy+2*tns)

	if MinimizerZXDebug && (NSteps < 20 || NSteps%500 == 0) {
		// on small steps zx cal should be ~1 and theta/sy ~0 if the energy scale is right
		println(NSteps, "zx cal", dE/(0.5*sy+tns), "theta/sy", theta/sy)
	}

	if ss == 0 || (MinimizerZXClip > 0 && math.Abs(theta) > MinimizerZXClip*math.Abs(sy)) {
		theta = 0
		MinimizerZXGated++
	} else {
		MinimizerZXUsed++
	}

	syHat := sy + theta
	if (syHat < 0) != (sy < 0) {
		MinimizerZXSignFlips++
	}

	var nom, div float64
	if NSteps%2 == 0 {
		nom, div = ss, syHat
	} else {
		nom, div = syHat, yy+2*theta*sy/ss+theta*theta/ss
	}
	if div != 0 {
		return float32(nom / div), true
	}
	return 0, false
}

// Initial step size from a strong-Wolfe (More-Thuente) search along the torque.
// Assumes uniform Msat and cell volume, so that E/(Ms*V) has gradient -t.
// k is the torque at the current M. M is restored exactly before returning.
func initialStepWolfe(k *data.Slice) float32 {
	m := M.Buffer()
	size := m.Size()

	tmax := float64(cuda.MaxVecNorm(k))
	if tmax == 0 {
		return 1e-4
	}

	wa := cuda.Buffer(3, size) // base point
	s := cuda.Buffer(3, size)  // search direction = torque
	g := cuda.Buffer(3, size)  // gradient = -torque
	defer cuda.Recycle(wa)
	defer cuda.Recycle(s)
	defer cuda.Recycle(g)

	data.Copy(wa, m)
	data.Copy(s, k)
	cuda.Madd2(g, k, k, -1, 0) // g = -k

	f0 := zxEnergy()

	// evalEG: cvsrch has already set M = wa + stp*s. Project back onto the sphere,
	// compute the torque there, turn it into a gradient, return the energy.
	evalEG := func(g *data.Slice) float64 {
		M.normalize()
		torqueFn(g)
		cuda.Madd2(g, g, g, -1, 0) // torque -> gradient
		return zxEnergy()
	}

	stp0 := math.Tan(MinimizerLSAngle) / tmax
	verbose := 0
	if MinimizerLSDebug {
		verbose = 2
	}
	_, stp, info := cvsrch(wa, f0, g, stp0, s, evalEG, verbose, MinimizerLSMaxAngle)

	data.Copy(m, wa) // restore M exactly; the real step is taken by Step()

	if info < 0 || info == 4 || math.IsNaN(stp) || stp <= 0 {
		if MinimizerLSDebug {
			println("wolfe: failed, info", info, "using start", stp0)
		}
		stp = stp0
	}

	// Convert the projection step to the retraction used by Step(),
	// matching the rotation angle of the max-torque cell.
	ang := math.Atan(stp * tmax)
	var h float64
	if MinimizerExact {
		h = ang / tmax
	} else {
		h = 2 * math.Tan(ang/2) / tmax // Cayley
	}

	if MinimizerLSDebug {
		println("wolfe: tmax", tmax, "stp0", stp0, "stp", stp, "angle(rad)", ang, "info", info, "h", h)
	}
	return float32(h)
}

func (mini *Minimizer) Step() {
	m := M.Buffer()
	size := m.Size()

	if mini.k == nil {
		mini.k = cuda.Buffer(3, size)
		torqueFn(mini.k)
		if MinimizerZX != 0 {
			mini.prevE = zxEnergy()
		}
		if MinimizerLS {
			mini.h = initialStepWolfe(mini.k)
		}
	}

	k := mini.k
	h := mini.h

	// save original magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	// make descent
	if MinimizerExact {
		cuda.MinimizeExp(m, m0, k, h)
	} else {
		cuda.Minimize(m, m0, k, h)
	}

	// calculate new torque for next step
	k0 := cuda.Buffer(3, size)
	defer cuda.Recycle(k0)
	data.Copy(k0, k)
	torqueFn(k)
	setMaxTorque(k) // report to user

	// energy at the new m (same state the new torque was computed from)
	var eNew float64
	if MinimizerZX != 0 {
		eNew = zxEnergy()
	}

	// Riemannian secant pair, must be computed before m0/k0 are overwritten below
	var rs, ry *data.Slice
	if MinimizerRiemann != 0 {
		rs = cuda.Buffer(3, size)
		ry = cuda.Buffer(3, size)
		defer cuda.Recycle(rs)
		defer cuda.Recycle(ry)
		cuda.RiemannSY(rs, ry, m0, m, k0, k)
	}

	// just to make the following readable
	dm := m0
	dk := k0

	// calculate step difference of m and k
	cuda.Madd2(dm, m, m0, 1., -1.)
	cuda.Madd2(dk, k, k0, -1., 1.) // reversed due to LLNoPrecess sign

	// get maxdiff and add to list (ambient dm, so the stopping criterion is unchanged)
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	setLastErr(mini.lastDm.Max()) // report maxDm to user

	// secant pair used for the step size
	sv, yv := dm, dk
	switch MinimizerRiemann {
	case 1:
		sv, yv = rs, ry
	case 2:
		sv, yv = rs, dk
	case 3:
		sv, yv = dm, ry
	}

	// adjust next time step
	var newH float32
	ok := false

	if MinimizerTwoStep {
		a := float32(math.Sqrt(float64(cuda.Dot(sv, sv))))

		if mini.prevDm != nil && a > 0 && mini.prevA > 0 {
			b := mini.prevA
			omega := a / (a + b)
			if omega <= 0.5 { // otherwise history is stale: use plain BB
				c1 := 1 + omega
				c2 := omega * omega / (1 - omega)
				r := cuda.Buffer(3, size)
				w := cuda.Buffer(3, size)
				cuda.Madd2(r, sv, mini.prevDm, c1, -c2)
				cuda.Madd2(w, yv, mini.prevDk, c1, -c2)
				newH, ok = bbStep(r, w)
				cuda.Recycle(r)
				cuda.Recycle(w)
			}
			if !ok {
				MinimizerFallbacks++
			}
		}

		// remember this step for the next two-step secant
		if mini.prevDm == nil {
			mini.prevDm = cuda.Buffer(3, size)
			mini.prevDk = cuda.Buffer(3, size)
		}
		data.Copy(mini.prevDm, sv)
		data.Copy(mini.prevDk, yv)
		mini.prevA = a
	}

	if !ok {
		useZX := MinimizerZX == 1 ||
			(MinimizerZX == 2 && NSteps%2 == 0) ||
			(MinimizerZX == 3 && NSteps%2 == 1)
		if useZX {
			newH, ok = bbStepZX(sv, yv, k, mini.prevE-eNew)
		} else {
			newH, ok = bbStep(sv, yv)
		}
	}
	if !ok {
		newH = 1e-4 // same as original zero-division fallback
	}

	// NaN/Inf guard, only active if MinimizerNaNGuard is set.
	if MinimizerNaNGuard && (math.IsNaN(float64(newH)) || math.IsInf(float64(newH), 0)) {
		newH = 1e-4
		MinimizerFallbacks++
	}

	// Optional cap on the rotation angle of the next step. Keeps the sign of h.
	// k now holds the torque the next step will use.
	if MinimizerMaxAngle > 0 {
		if tn := float64(cuda.MaxVecNorm(k)); tn > 0 {
			lim := float32(MinimizerMaxAngle / tn)
			if newH > lim {
				newH = lim
				MinimizerAngleClamps++
			} else if newH < -lim {
				newH = -lim
				MinimizerAngleClamps++
			}
		}
	}

	mini.h = newH
	if newH < 0 {
		MinimizerNegSteps++
	}

	if MinimizerZX != 0 {
		mini.prevE = eNew
	}

	M.normalize()

	// as a convention, time does not advance during relax
	NSteps++
}

func (mini *Minimizer) Free() {
	if mini.k != nil {
		mini.k.Free()
		mini.k = nil
	}
	if mini.prevDm != nil {
		mini.prevDm.Free()
		mini.prevDk.Free()
		mini.prevDm = nil
		mini.prevDk = nil
	}
}

// helper function that returns false if the wall clock time limit is exceeded. If the wall-clock time is negative, this function always returns true.
func WallclockTimer(start time.Time, WallClockTime float64) bool {
	if WallClockTime < 0 {
		return true
	}
	if WallClockTime == 0 {
		return false
	}
	return time.Since(start) < time.Duration(WallClockTime*float64(time.Second))
}

func Minimize() bool {

	// if wall-clock time is zero, skip minimization entirely (zero steps), and don't change any settings
	MinimizeConverged = false
	TimerStart := time.Now()
	if MinimizeWallClockTime == 0 {
		MinimizeConverged = false
		return MinimizeConverged
	}

	Refer("exl2014")
	SanityCheck()

	MinimizerFallbacks = 0
	MinimizerNegSteps = 0
	MinimizerZXUsed = 0
	MinimizerZXGated = 0
	MinimizerZXSignFlips = 0
	MinimizerAngleClamps = 0

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
		h:      1e-4,
		k:      nil,
		lastDm: FifoRing(DmSamples)}
	stepper = &mini

	cond := func() bool {
		return (mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm) && WallclockTimer(TimerStart, MinimizeWallClockTime)
	}

	RunWhile(cond)
	pause = true
	// if the loop ended because of convergence, then MinimizeConverged is true. If the loop ended because of wall-clock time, then MinimizeConverged is false.
	MinimizeConverged = !(mini.lastDm.count < DmSamples || mini.lastDm.Max() > StopMaxDm)
	stepper.Free()
	return MinimizeConverged
}
