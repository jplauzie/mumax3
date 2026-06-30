// engine/cgevolve.go
// Conjugate gradient energy minimizer for mumax3.
// Ported from OOMMF cgevolve.cc (Michael J. Donahue, NIST).
//
// Algorithm summary
// -----------------
// Each call to (*CGEvolve).Step() performs exactly one energy evaluation
// (one FillBracket call), mirroring OOMMF's TryStep.  The outer loop
// (CGMinimize → RunWhile) keeps calling Step() until convergence.
//
// Within a line search the sequence is:
//   1. SetBasePoint – compute (conjugate) gradient direction.
//   2. FindBracketStep (repeated) – expand until minimum is bracketed.
//   3. FindLineMinimumStep (repeated) – compress bracket to find minimum.
//   4. Return to 1 with the new best point as the next base point.
//
// Convergence criterion: max|mxHxm| < CGMinimizerStop (same units as
// mumax3's existing Minimize).

package engine

import (
	"fmt"
	"math"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// ---------------------------------------------------------------------------
// Public parameters (can be set from mumax3 scripts)
// ---------------------------------------------------------------------------

var (
	CGMinimizerStop          float64 = 1e-6
	CGMinimizerSamples       int     = 10
	CGMinimizerMethod        string  = "Fletcher-Reeves" // or "Polak-Ribiere"
	CGMinimizerWallClockTime float64 = -1.0
	CGMinimizerConverged     bool
	dirMaxMagRaw             float64
	dirNormRaw               float64
)

func init() {
	DeclFunc("CGMinimize", CGMinimize,
		"Conjugate-gradient energy minimizer (OOMMF port). Returns true on convergence.")
	DeclVar("CGMinimizerStop", &CGMinimizerStop,
		"Stopping max|mxHxm| for CGMinimize")
	DeclVar("CGMinimizerSamples", &CGMinimizerSamples,
		"Convergence window (number of max|mxHxm| samples) for CGMinimize")
	DeclVar("CGMinimizerMethod", &CGMinimizerMethod,
		"CG method: Fletcher-Reeves or Polak-Ribiere")
	DeclVar("CGStage", &CGStage,
		"Increment to force a CG direction reset on the next step (e.g. after changing applied field mid-minimization)")
}

// CGStage is a stage counter mirroring OOMMF's stage-change detection.
// A fresh CGMinimize call always starts with basept.valid=false so the
// direction resets automatically; CGStage is only needed when reusing a
// persistent CGEvolve across parameter changes.
var CGStage int

// ---------------------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------------------

const (
	mu0 = 4e-7 * math.Pi

	// Default bracket step limits (degrees → tan, see constructor)
	cgDefaultMinStepDeg   = 0.05
	cgDefaultMaxStepDeg   = 10.0
	cgDefaultAnglePrecDeg = 1.0     // line_minimum_angle_precision
	cgDefaultRelMinSpan   = 1.0e-11 // line_minimum_relwidth

	// CG direction reset parameters
	cgDefaultGradResetCount   = 5000
	cgDefaultGradResetAngle   = 87.5 // degrees
	cgDefaultKludgeAngle      = 89.2 // degrees
	cgDefaultGradResetWgt     = 31.0 / 32.0
	cgDefaultGradResetTrigger = 0.5
	cgOffsetEpsilon           = 1e-7 // smallest offset that survives float32 CgSpinStep
	// without saturating/flatlining (empirically determined;
	// OOMMF's OC_REAL8m_EPSILON equivalent doesn't transfer
	// directly because direction is normalized to O(1) here
	// and CgSpinStep runs in float32, unlike OOMMF's raw-
	// offset/double-precision convention)
)

// ---------------------------------------------------------------------------
// cgBracket: one point along the line search
// ---------------------------------------------------------------------------

type cgBracket struct {
	spin     *data.Slice // GPU: magnetization at this offset
	mxHxm    *data.Slice // GPU: effective torque at this offset
	totalE   float64     // absolute total energy (J)
	offset   float64     // parameter along basept.direction; -1 = uninitialised
	Ep       float64     // dE/d(offset), scaled by -mu0*V
	gradNorm float64     // cellVol * sqrt(Σ |mxHxm|² * scale_adj²)
	Eerr     float64     // estimated |E| error
}

func newCGBracket(sz [3]int) *cgBracket {
	return &cgBracket{
		spin:   cuda.Buffer(3, sz),
		mxHxm:  cuda.Buffer(3, sz),
		offset: -1,
	}
}

func (b *cgBracket) free() {
	if b == nil {
		return
	}
	if b.spin != nil {
		cuda.Recycle(b.spin)
		b.spin = nil
	}
	if b.mxHxm != nil {
		cuda.Recycle(b.mxHxm)
		b.mxHxm = nil
	}
}

// relE returns the energy of b relative to an absolute reference.
func (b *cgBracket) relE(refTotalE float64) float64 { return b.totalE - refTotalE }

// ---------------------------------------------------------------------------
// cgBasept: the start of the current line search
// ---------------------------------------------------------------------------

type cgBasept struct {
	valid        bool
	stage        int
	totalEnergy  float64
	direction    *data.Slice // GPU: current search direction (tangent to sphere)
	dirMaxMag    float64     // sup-norm of direction
	dirNorm      float64     // L2-norm of direction
	prevMxHxm    *data.Slice // GPU: previous mxHxm (Polak-Ribière only; nil for FR)
	gSumSq       float64     // ||P·g_{n-1}||²  – gamma denominator from last cycle
	Ep           float64     // dE/dt at t = 0  (must be ≤ 0)
	method       int         // 0 = Fletcher-Reeves, 1 = Polak-Ribière
	dirMaxMagRaw float64
}

const (
	methodFR = 0
	methodPR = 1
)

// ---------------------------------------------------------------------------
// CGEvolve: the full conjugate-gradient minimizer
// ---------------------------------------------------------------------------

type CGEvolve struct {
	// ---- user parameters (from defaults, adjustable before calling) ----
	minstep        float64 // bracket.minstep (as tan(angle))
	maxstep        float64 // bracket.maxstep (as tan(angle))
	anglePrecision float64 // sin(line_minimum_angle_precision)
	relMinSpan     float64

	gradResetCount          int
	gradResetAngleCotangent float64
	gradResetWgt            float64
	gradResetTrigger        float64
	kludgeAdjustAngleCos    float64

	// ---- per-run state -------------------------------------------------
	basept cgBasept

	// brackets
	left, right *cgBracket // left and right brackets for current line search
	extra       *cgBracket // spare bracket (extra_bracket in OOMMF)

	minBracketed     bool
	minFound         bool
	badEdata         bool
	weakBracketCount int

	scaledMinstep, scaledMaxstep float64
	startStep, stopSpan          float64
	lastMinReductionRatio        float64
	nextToLastMinReductionRatio  float64

	// best point (always one of left/right)
	bestBracket   *cgBracket
	bestIsLineMin bool

	// gradient reset scoring
	gradResetScore float64
	sumErrEstimate float64

	// per-cell Ms*V GPU buffer (rebuilt on mesh change)
	msv      *data.Slice // normalized: Ms*V / msvRef
	msvRef   float64     // reference scale, so msv[i]*msvRef = actual Ms[i]*V
	meshSize [3]int

	// diagnostics / convergence
	lastMaxMxHxm    fifoRing
	energyCalcCount int
	bracketCount    int
	lineMinCount    int
	cycleCount      int
	cycleSubCount   int
	conjCycleCount  int
}

func (cg *CGEvolve) normalizeDirection(dirMaxMagRaw float64) {
	if dirMaxMagRaw <= 0 {
		return
	}
	cuda.Madd2(cg.basept.direction, cg.basept.direction, cg.basept.direction,
		float32(1.0/dirMaxMagRaw), 0)
}

// newCGEvolve creates a CGEvolve with OOMMF-compatible default parameters.
func newCGEvolve() *CGEvolve {
	cg := &CGEvolve{
		gradResetWgt:     cgDefaultGradResetWgt,
		gradResetTrigger: cgDefaultGradResetTrigger,
		gradResetCount:   cgDefaultGradResetCount,
	}
	minAng := cgDefaultMinStepDeg * math.Pi / 180
	maxAng := cgDefaultMaxStepDeg * math.Pi / 180
	cg.minstep = math.Tan(minAng)
	cg.maxstep = math.Tan(maxAng)
	cg.anglePrecision = math.Sin(cgDefaultAnglePrecDeg * math.Pi / 180)
	cg.relMinSpan = cgDefaultRelMinSpan
	cg.gradResetAngleCotangent = math.Tan(
		math.Abs(90-cgDefaultGradResetAngle) * math.Pi / 180)
	cg.kludgeAdjustAngleCos = math.Cos(cgDefaultKludgeAngle * math.Pi / 180)

	if CGMinimizerMethod == "Polak-Ribiere" {
		cg.basept.method = methodPR
	}
	return cg
}

// Free releases all GPU buffers.
func (cg *CGEvolve) Free() {
	cg.left.free()
	cg.right.free()
	cg.extra.free()
	if cg.basept.direction != nil {
		cuda.Recycle(cg.basept.direction)
	}
	if cg.basept.prevMxHxm != nil {
		cuda.Recycle(cg.basept.prevMxHxm)
	}
	if cg.msv != nil {
		cuda.Recycle(cg.msv)
	}
}

// ---------------------------------------------------------------------------
// Step: one energy evaluation (mirrors OOMMF TryStep)
// ---------------------------------------------------------------------------

// Step advances the CG line search by one energy evaluation.
// After a complete line minimum is found it resets the base point so the
// next call starts a new line search.
func (cg *CGEvolve) Step() {
	sz := M.Buffer().Size()
	cg.ensureMsV(sz)
	if !cg.basept.valid || cg.minFound {
		cg.setBasePoint()
	}
	if !cg.minBracketed {
		cg.findBracketStep()
	} else if !cg.minFound {
		cg.findLineMinimumStep()
	}
	if cg.minFound && cg.bestBracket.offset == 0 {
		if cg.cycleSubCount == 0 {
			cg.nudgeBestpt()
		} else {
			cg.basept.valid = false
			cg.setBasePoint()
		}
	}
	if cg.bestBracket != nil && cg.bestBracket.spin != nil {
		data.Copy(M.Buffer(), cg.bestBracket.spin)
		M.normalize()
	}
	if cg.bestBracket != nil && cg.bestBracket.mxHxm != nil {
		maxTorque := float64(cuda.MaxVecNorm(cg.bestBracket.mxHxm))
		cg.lastMaxMxHxm.Add(maxTorque)
		setLastErr(cg.lastMaxMxHxm.Max())
		setMaxTorque(cg.bestBracket.mxHxm)

		// DEBUG: trace every single Step() call's outcome
		mHost := M.Buffer().HostCopy()
		fmt.Printf("Step #%d: bestBracket.offset=%e bestBracket.Ep=%e maxTorque=%e | m[tilt or repr cell]=(%e,%e,%e) | minFound=%v minBracketed=%v basept.valid=%v\n",
			NSteps, cg.bestBracket.offset, cg.bestBracket.Ep, maxTorque,
			mHost.Vectors()[0][0][2][2], mHost.Vectors()[1][0][2][2], mHost.Vectors()[2][0][2][2],
			cg.minFound, cg.minBracketed, cg.basept.valid)
	}
	NSteps++
}

// ---------------------------------------------------------------------------
// ensureMsV: rebuild Ms*V buffer when mesh changes
// ---------------------------------------------------------------------------

func (cg *CGEvolve) ensureMsV(sz [3]int) {
	if cg.msv != nil && sz == cg.meshSize {
		return
	}
	if cg.msv != nil {
		cuda.Recycle(cg.msv)
	}
	cs := Mesh().CellSize()
	vol := float32(cs[0] * cs[1] * cs[2])
	ms := Msat.MSlice()
	defer ms.Recycle()
	cg.msv = cuda.Buffer(1, sz)
	cg.msvRef = float64(cuda.CgBuildMsV(cg.msv, ms, vol))
	cg.meshSize = sz
	n := float64(Mesh().NCell())
	cg.sumErrEstimate = 1e-15 * math.Sqrt(n) // OC_REAL8m_EPSILON * sqrt(size)
}

// ---------------------------------------------------------------------------
// cellVol / totalEnergy helpers
// ---------------------------------------------------------------------------

func cgCellVol() float64 {
	cs := Mesh().CellSize()
	return cs[0] * cs[1] * cs[2]
}

// cgTorqueFn computes mxHxm into dst without aliasing H_eff and torque buffers.
// SetLLTorque reuses dst for both H_eff and torque output, which violates
// __restrict__ in LLNoPrecess. We avoid this by using a separate H_eff buffer.
func cgTorqueFn(dst *data.Slice) {
	heff := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(heff)
	SetEffectiveField(heff)
	if Msat.hasZero() {
		cuda.ZeroMaskInv(heff, Msat.gpuLUT1(), regions.Gpu())
	}

	hHost := heff.HostCopy()
	fmt.Printf("H_eff: tilt=(%e, %e, %e) nbr=(%e, %e, %e)\n",
		hHost.Vectors()[0][0][2][2], hHost.Vectors()[1][0][2][2], hHost.Vectors()[2][0][2][2],
		hHost.Vectors()[0][0][2][1], hHost.Vectors()[1][0][2][1], hHost.Vectors()[2][0][2][1])

	cuda.LLNoPrecess(dst, M.Buffer(), heff)

	t := dst.HostCopy()
	fmt.Printf("mxHxm: tilt=(%e, %e, %e) nbr=(%e, %e, %e)\n",
		t.Vectors()[0][0][2][2], t.Vectors()[1][0][2][2], t.Vectors()[2][0][2][2],
		t.Vectors()[0][0][2][1], t.Vectors()[1][0][2][1], t.Vectors()[2][0][2][1])
	NEvals++
}

// cgGetEnergyAndTorque writes the current effective torque into dst and
// returns the total energy in Joules.
func cgGetEnergyAndTorque(dst *data.Slice) float64 {
	cgTorqueFn(dst) // compute mxHxm into dst
	h := dst.HostCopy()
	fmt.Printf("H_eff: (%e, %e, %e)\n", h.Vectors()[0][0][0][0],
		h.Vectors()[1][0][0][0],
		h.Vectors()[2][0][0][0])
	return GetTotalEnergy() // sum all enabled energy terms
}

// ---------------------------------------------------------------------------
// fillBracket: evaluate E, Ep, gradNorm at a given line offset
// ---------------------------------------------------------------------------

// fillBracket computes the spin, torque, energy, and line-derivative at
// the given offset along basept.direction, starting from bestBracket.
// Results are written into dst.
func (cg *CGEvolve) fillBracket(offset float64, dst *cgBracket) {
	fmt.Printf("fillBracket offset=%e\n", offset)
	cg.energyCalcCount++
	cg.energyCalcCount++
	best := cg.bestBracket
	tsq := float32(offset * offset)
	dscale := float32(offset)

	// DEBUG: both cells' spin going into CgSpinStep, plus the direction
	// being applied (so we can check it's the same direction buffer
	// setBasePoint just printed, not something stale).
	{
		bestHostPre := best.spin.HostCopy()
		dirHostPre := cg.basept.direction.HostCopy()
		fmt.Printf("CgSpinStep INPUT: best[0]=(%e,%e,%e) best[1]=(%e,%e,%e) | dir[0]=(%e,%e,%e) dir[1]=(%e,%e,%e) | tsq=%e dscale=%e\n",
			bestHostPre.Vectors()[0][0][0][0], bestHostPre.Vectors()[1][0][0][0], bestHostPre.Vectors()[2][0][0][0],
			bestHostPre.Vectors()[0][0][0][1], bestHostPre.Vectors()[1][0][0][1], bestHostPre.Vectors()[2][0][0][1],
			dirHostPre.Vectors()[0][0][0][0], dirHostPre.Vectors()[1][0][0][0], dirHostPre.Vectors()[2][0][0][0],
			dirHostPre.Vectors()[0][0][0][1], dirHostPre.Vectors()[1][0][0][1], dirHostPre.Vectors()[2][0][0][1],
			tsq, dscale)
	}

	cuda.CgSpinStep(dst.spin, best.spin, cg.basept.direction,
		tsq, dscale)

	// DEBUG: measure actual rotation vs predicted, both cells
	bestHost := best.spin.HostCopy()
	newHost := dst.spin.HostCopy()
	bx, by, bz := bestHost.Vectors()[0][0][0][0], bestHost.Vectors()[1][0][0][0], bestHost.Vectors()[2][0][0][0]
	nx, ny, nz := newHost.Vectors()[0][0][0][0], newHost.Vectors()[1][0][0][0], newHost.Vectors()[2][0][0][0]
	dot := bx*nx + by*ny + bz*nz
	achievedAngle := math.Acos(math.Min(1, math.Max(-1, float64(dot))))

	bx1, by1, bz1 := bestHost.Vectors()[0][0][0][1], bestHost.Vectors()[1][0][0][1], bestHost.Vectors()[2][0][0][1]
	nx1, ny1, nz1 := newHost.Vectors()[0][0][0][1], newHost.Vectors()[1][0][0][1], newHost.Vectors()[2][0][0][1]
	dot1 := bx1*nx1 + by1*ny1 + bz1*nz1
	achievedAngle1 := math.Acos(math.Min(1, math.Max(-1, float64(dot1))))

	// Cross-cell agreement check: how aligned are the two cells with each
	// other, before and after this step. This is the key diagnostic for
	// the domain-patchwork symptom -- if cellDot drops (cells diverging)
	// after a step that should be making them agree, that's the bug.
	cellDotBefore := float64(bx*bx1 + by*by1 + bz*bz1)
	cellDotAfter := float64(nx*nx1 + ny*ny1 + nz*nz1)

	fmt.Printf("offset=%e  achievedAngle[0](rad)=%e  achievedAngle[1](rad)=%e  offset*dirMaxMag(corrected)=%e  cellDot(m0.m1) before=%e after=%e\n",
		offset, achievedAngle, achievedAngle1, offset*cg.basept.dirMaxMag, cellDotBefore, cellDotAfter)

	saved := cuda.Buffer(3, M.Buffer().Size())
	defer cuda.Recycle(saved)
	data.Copy(saved, M.Buffer())
	data.Copy(M.Buffer(), dst.spin)
	M.normalize()
	dst.totalE = cgGetEnergyAndTorque(dst.mxHxm)
	data.Copy(M.Buffer(), saved)
	dst.offset = offset
	epSum, gsqSum := cuda.CgEpAndGradSq(
		dst.mxHxm, cg.basept.direction, cg.msv, tsq)
	dst.Ep = -epSum * cg.msvRef
	dst.gradNorm = math.Sqrt(gsqSum) * cg.msvRef
	dst.Eerr = math.Abs(dst.totalE) * 8e-16
}

// ---------------------------------------------------------------------------
// updateBrackets: insert a new bracket point, keep minimum bracketed
// ---------------------------------------------------------------------------

// updateBrackets inserts newB into the left/right bracket pair, replacing
// whichever endpoint it is best to discard in order to keep the minimum
// bracketed. bestBracket is updated if newB (or the surviving endpoint) is
// better than the current best.
//
// Returns the discarded bracket (ready for re-use as extra_bracket).
func (cg *CGEvolve) updateBrackets(newB *cgBracket, forceBest bool) (discarded *cgBracket) {
	fmt.Printf("updateBrackets ENTRY: newB.offset=%e newB.Ep=%e left.offset=%e left.Ep=%e right.offset=%e right.Ep=%e\n",
		newB.offset, newB.Ep,
		cg.left.offset, cg.left.Ep,
		cg.right.offset, cg.right.Ep)
	slack := cg.estimateEnergySlack()
	bref := cg.bestBracket.totalE // absolute energy of current best

	// Helper: relative energy w.r.t. current best.
	relE := func(b *cgBracket) float64 { return b.totalE - bref }

	// ---- Case 1: right bracket not yet initialised ----
	if cg.right == nil || cg.right.offset < 0 {
		// bestBracket currently points to left (the only valid bracket).
		// We're replacing right with newB, left is untouched — no rescue needed.
		discarded = cg.right
		cg.right = newB
		cg.updateBestpt(forceBest, newB, slack)
		return
	}

	// ---- Case 2: newB is to the right of right bracket (expand) ----
	if newB.offset > cg.right.offset {
		rE := relE(cg.right)
		lE := relE(cg.left)
		if rE <= lE+slack && cg.right.Ep < 0 {
			if rE <= lE-slack ||
				(rE <= lE+slack && math.Abs(cg.right.Ep) <= math.Abs(cg.left.Ep)) {
				// Discarding left — rescue bestBracket if it pointed there.
				if cg.bestBracket == cg.left {
					cg.bestBracket = cg.right
				}
				discarded = cg.left
				cg.left = cg.right
			} else {
				// Discarding right — rescue bestBracket if it pointed there.
				if cg.bestBracket == cg.right {
					cg.bestBracket = cg.left
				}
				discarded = cg.right
			}
		} else {
			// Discarding right — rescue bestBracket if it pointed there.
			if cg.bestBracket == cg.right {
				cg.bestBracket = cg.left
			}
			discarded = cg.right
		}
		cg.right = newB
		cg.updateBestpt(forceBest, newB, slack)
		return
	}

	// ---- Case 3: newB is between left and right (minimum is bracketed) ----
	keep := 0 // -1 = keep left, +1 = keep right
	if newB.Ep >= 0 {
		keep = -1 // new point has Ep≥0, so min is in [left, new] → keep left
	} else if cg.right.Ep >= 0 {
		keep = 1 // right has Ep≥0 → keep right
	} else {
		// All Ep<0; use E data to decide.
		lE := relE(cg.left)
		nE := relE(newB)
		rE := relE(cg.right)
		if lE >= nE {
			keep = 1
		} else if nE >= rE {
			keep = -1
		} else {
			// Both sub-intervals might contain a minimum.

			lBad := cg.badPrecisionTest(cg.left, newB, slack)
			rBad := cg.badPrecisionTest(newB, cg.right, slack)
			switch {
			case lBad && !rBad:
				keep = 1
			case !lBad && rBad:
				keep = -1
			case nE-lE > rE-nE:
				keep = -1
				if lBad {
					cg.badEdata = true
				}
			default:
				keep = 1
				if rBad {
					cg.badEdata = true
				}
			}
		}
	}

	// If bestBracket points to the bracket we're about to discard,
	// rescue it: the surviving bracket is the new best reference.
	if keep == -1 {
		// discarding right → if bestBracket was right, it's about to be lost
		if cg.bestBracket == cg.right {
			cg.bestBracket = cg.left // preserve best as left before overwriting right
		}
		discarded = cg.right
		cg.right = newB
	} else {
		// discarding left → if bestBracket was left, it's about to be lost
		if cg.bestBracket == cg.left {
			cg.bestBracket = cg.right // preserve best as right before overwriting left
		}
		discarded = cg.left
		cg.left = newB
	}
	cg.updateBestpt(forceBest, newB, slack)
	return
}

// updateBestpt picks the better of left/right as bestBracket,
// or forces it to forcedB if forceBest is true.
func (cg *CGEvolve) updateBestpt(forceBest bool, forcedB *cgBracket, slack float64) {
	if forceBest {
		cg.bestBracket = forcedB
		fmt.Printf("updateBestpt: forceBest=true -> bestBracket=forcedB offset=%e Ep=%e totalE=%e\n",
			cg.bestBracket.offset, cg.bestBracket.Ep, cg.bestBracket.totalE)
		return
	}
	bref := cg.bestBracket.totalE
	lE := cg.left.totalE - bref
	rE := cg.right.totalE - bref
	if rE < lE-slack || (rE < lE+slack && math.Abs(cg.right.Ep) < math.Abs(cg.left.Ep)) {
		cg.bestBracket = cg.right
	} else {
		cg.bestBracket = cg.left
	}
	fmt.Printf("updateBestpt: forceBest=false lE=%e rE=%e slack=%e -> chose=%s offset=%e Ep=%e totalE=%e\n",
		lE, rE, slack,
		func() string {
			if cg.bestBracket == cg.right {
				return "right"
			}
			return "left"
		}(),
		cg.bestBracket.offset, cg.bestBracket.Ep, cg.bestBracket.totalE)
}

// ---------------------------------------------------------------------------
// estimateEnergySlack
// ---------------------------------------------------------------------------

func (cg *CGEvolve) estimateEnergySlack() float64 {
	lErr := 0.0
	rErr := 0.0
	if cg.left != nil {
		lErr = cg.left.Eerr
	}
	if cg.right != nil {
		rErr = cg.right.Eerr
	}
	edelta := 0.0
	if cg.right != nil && cg.right.offset > 0 {
		edelta = (math.Abs(cg.basept.Ep) + math.Abs(cg.left.Ep) +
			math.Abs(cg.right.Ep)) * 0.5 * cg.right.offset
	}
	slack := lErr + rErr + edelta*1e-15

	// Floor at float32 noise in absolute totalE from GPU reduction
	const float32Eps = 1.192093e-07
	absEnergyNoise := float32Eps * math.Abs(cg.left.totalE)
	if absEnergyNoise > slack {
		slack = absEnergyNoise
	}
	fmt.Printf("estimateEnergySlack: lErr=%e rErr=%e edelta=%e edelta*1e-15=%e -> slack=%e  (leftTotalE=%e rightTotalE=%e basept.Ep=%e)\n",
		lErr, rErr, edelta, edelta*1e-15, slack,
		func() float64 {
			if cg.left != nil {
				return cg.left.totalE
			}
			return math.NaN()
		}(),
		func() float64 {
			if cg.right != nil {
				return cg.right.totalE
			}
			return math.NaN()
		}(),
		cg.basept.Ep)
	return slack
}

// badPrecisionTest returns true if the numeric data in [l,r] looks unreliable.
func (cg *CGEvolve) badPrecisionTest(l, r *cgBracket, slack float64) bool {
	span := r.offset - l.offset
	lEp := l.Ep * span
	rEp := r.Ep * span

	cond1 := span <= 256*cg.stopSpan
	cond2 := math.Abs(rEp-lEp) < math.Abs(lEp)/16
	cond3 := math.Abs(lEp) < slack
	result := cond1 && cond2 && cond3

	fmt.Printf("badPrecisionTest: span=%e stopSpan=%e 256*stopSpan=%e cond1(span<=256*stopSpan)=%v | lEp=%e rEp=%e |rEp-lEp|=%e |lEp|/16=%e cond2=%v | slack=%e |lEp|=%e cond3(|lEp|<slack)=%v || RESULT=%v\n",
		span, cg.stopSpan, 256*cg.stopSpan, cond1,
		lEp, rEp, math.Abs(rEp-lEp), math.Abs(lEp)/16, cond2,
		slack, math.Abs(lEp), cond3, result)

	return result
}

// ---------------------------------------------------------------------------
// findBracketStep: expand the bracket until the minimum is enclosed
// ---------------------------------------------------------------------------

func (cg *CGEvolve) findBracketStep() {
	fmt.Printf("findBracketStep ENTRY: scaledMinstep=%e scaledMaxstep=%e startStep=%e leftOffset=%e rightOffset=%e leftEp=%e\n",
		cg.scaledMinstep, cg.scaledMaxstep, cg.startStep, cg.left.offset,
		func() float64 {
			if cg.right != nil {
				return cg.right.offset
			}
			return -1
		}(), cg.left.Ep)

	rightOffset := 0.0
	if cg.right != nil && cg.right.offset >= 0 {
		rightOffset = cg.right.offset
	}
	if cg.left.Ep == 0 {
		if cg.right == nil || cg.right.offset < 0 {
			cg.fillBracket(1e-15, cg.extra)
			cg.extra = cg.updateBrackets(cg.extra, false)
		} else {
			cg.minBracketed = true
		}
		return
	}
	cg.bracketCount++
	slack := cg.estimateEnergySlack()
	var offset float64
	h := rightOffset - cg.left.offset
	maxOff := math.Max(cg.left.offset, cg.scaledMaxstep)
	minOff := math.Min(math.Max(cg.startStep, 2*rightOffset), maxOff)

	fmt.Printf("findBracketStep CALC: h=%e minOff=%e maxOff=%e rightOffset=%e leftOffset=%e\n",
		h, minOff, maxOff, rightOffset, cg.left.offset)

	if h <= 0 {
		offset = minOff
		fmt.Printf("findBracketStep BRANCH: h<=0, offset=minOff=%e\n", offset)
	} else {
		wgt := 0.5
		if cg.badEdata {
			wgt = 0
		}
		quadEst := estimateQuadraticMinimum(wgt, h,
			cg.left.totalE-cg.bestBracket.totalE,
			cg.right.totalE-cg.bestBracket.totalE,
			cg.left.Ep, cg.right.Ep)
		est := cg.left.offset + 1.75*quadEst
		fmt.Printf("findBracketStep BRANCH: h>0, quadEst=%e est_before_clamp=%e\n", quadEst, est)
		offset = math.Max(minOff, math.Min(est, maxOff))
		fmt.Printf("findBracketStep BRANCH: offset_after_minmax_clamp=%e\n", offset)
	}

	offsetAfterQuadratic := offset
	fmt.Printf("findBracketStep CHECKPOINT1: offsetAfterQuadratic=%e\n", offsetAfterQuadratic)

	// Clamp to avoid floating-point zero span.
	if offset <= rightOffset {
		fmt.Printf("findBracketStep ZEROSPAN: offset(%e) <= rightOffset(%e), entering zero-span fixup\n", offset, rightOffset)
		if rightOffset > 0 {
			offset = rightOffset * (1 + 16*1e-15)
			fmt.Printf("findBracketStep ZEROSPAN: rightOffset>0 branch, offset=%e\n", offset)
		} else if cg.scaledMaxstep > 0 {
			offset = cg.scaledMaxstep
			fmt.Printf("findBracketStep ZEROSPAN: scaledMaxstep>0 branch, offset=%e\n", offset)
		} else {
			offset = 1e-15
			fmt.Printf("findBracketStep ZEROSPAN: punt branch, offset=%e\n", offset)
		}
	}

	offsetFinal := offset
	fmt.Printf("findBracketStep CHECKPOINT2: offsetFinal=%e (about to call fillBracket)\n", offsetFinal)

	cg.fillBracket(offset, cg.extra)
	cg.extra = cg.updateBrackets(cg.extra, false)
	// Classify result.
	rE := cg.right.totalE - cg.bestBracket.totalE
	lE := cg.left.totalE - cg.bestBracket.totalE
	if (cg.badEdata || rE <= lE+slack) && cg.right.Ep < 0 {
		cg.minBracketed = false

		cg.stopSpan = 0
		fmt.Printf("findBracketStep SET stopSpan: relMinSpan=%e right.offset=%e -> stopSpan=%e\n",
			cg.relMinSpan, cg.right.offset, cg.stopSpan)
		if cg.right.offset >= cg.scaledMaxstep {
			// Cannot bracket within allowed range; accept bestBracket.
			cg.minBracketed = true
			cg.minFound = true
		}
	} else {
		cg.minBracketed = true

		cg.stopSpan = cg.relMinSpan * cg.right.offset
		fmt.Printf("findBracketStep SET stopSpan: relMinSpan=%e right.offset=%e -> stopSpan=%e\n",
			cg.relMinSpan, cg.right.offset, cg.stopSpan)
		if cg.stopSpan*cg.basept.dirMaxMag < 4*1e-15 {
			cg.stopSpan = 4 * 1e-15 / cg.basept.dirMaxMag
		}
	}
	_ = slack
	fmt.Printf("findBracketStep CLASSIFY: rightE=%e leftE=%e slack=%e rightEp=%e minBracketed=%v minFound=%v badEdata=%v\n",
		rE, lE, slack, cg.right.Ep, cg.minBracketed, cg.minFound, cg.badEdata)
}

// ---------------------------------------------------------------------------
// findLineMinimumStep: compress the bracket to the line minimum
// ---------------------------------------------------------------------------

func (cg *CGEvolve) findLineMinimumStep() {
	fmt.Printf("findLineMinimumStep ENTRY\n")
	span := cg.right.offset - cg.left.offset
	slack := cg.estimateEnergySlack()

	const float32Epsilon = 1.1920929e-7
	precFloor := float32Epsilon / cg.basept.dirMaxMag // = 1.19e-7 since dirMaxMag=1

	nudge := precFloor // initial value, replacing cgOffsetEpsilon

	if nudge >= 0.125*span && cg.right.Ep > cg.left.Ep*(1-1e-15) {
		clamped := 0.125 * span
		if clamped > precFloor {
			nudge = clamped // only clamp if result stays above precision floor
		}
		// else: leave nudge at precFloor
	}

	// Check convergence.
	convergenceRHS := cg.bestBracket.gradNorm * cg.basept.dirNorm *
		cg.anglePrecision * (1 + 2*cg.sumErrEstimate)
	converged := math.Abs(cg.bestBracket.Ep) < convergenceRHS
	fmt.Printf("CONVERGE_CHECK: |bestEp|=%e RHS=%e ratio=%e gradNorm=%e dirNorm=%e dirMaxMagRaw=%e\n",
		math.Abs(cg.bestBracket.Ep),
		cg.bestBracket.gradNorm*cg.basept.dirNorm*cg.anglePrecision*(1+2*cg.sumErrEstimate),
		math.Abs(cg.bestBracket.Ep)/(cg.bestBracket.gradNorm*cg.basept.dirNorm*cg.anglePrecision*(1+2*cg.sumErrEstimate)),
		cg.bestBracket.gradNorm, cg.basept.dirNorm, cg.basept.dirMaxMagRaw)

	fmt.Printf("CONVERGE_GATES: converged=%v bestEp_gt_baseptEp=%v bestEp_zero=%v offset_zero=%v span_le_stopSpan=%v nudge_ge_span=%v ratio=%.6e\n",
		converged,
		cg.bestBracket.Ep > cg.basept.Ep,
		cg.bestBracket.Ep == 0,
		cg.bestBracket.offset == 0,
		span <= cg.stopSpan,
		nudge >= span,
		math.Abs(cg.bestBracket.Ep)/convergenceRHS)

	convergedAndBetter := converged &&
		(cg.bestBracket.Ep == 0 ||
			cg.bestBracket.Ep > cg.basept.Ep ||
			cg.bestBracket.offset == 0)

	if convergedAndBetter &&
		(cg.bestBracket.Ep == 0 ||
			cg.bestBracket.Ep > cg.basept.Ep || // if bestEp > baseptEp, skip span check
			span <= cg.stopSpan ||
			nudge >= span) {
		cg.minFound = true
		cg.bestIsLineMin = true
		cg.lastMinReductionRatio = 0
		cg.nextToLastMinReductionRatio = 0
		return
	}

	if cg.left.Ep >= 0 || nudge >= span*(1-1e-15) || cg.right.Ep == 0 {
		cg.bestIsLineMin = nudge >= span*(1-1e-15)
		cg.minFound = true
		cg.lastMinReductionRatio = 0
		cg.nextToLastMinReductionRatio = 0
		return
	}

	fmt.Printf("PRE-BAD: ... converged=%v convergenceRHS=%e bestEp=%e gradNorm=%e dirNorm=%e anglePrecision=%e sumErrEstimate=%e\n",
		converged, convergenceRHS,
		cg.bestBracket.Ep, cg.bestBracket.gradNorm, cg.basept.dirNorm,
		cg.anglePrecision, cg.sumErrEstimate)

	if cg.badPrecisionTest(cg.left, cg.right, slack) {
		cg.minFound = true
		cg.badEdata = true
		cg.lastMinReductionRatio = 0
		cg.nextToLastMinReductionRatio = 0
		return
	}

	// Choose interpolation point.
	lEp := cg.left.Ep * span
	rEp := cg.right.Ep * span
	Ediff := (cg.right.totalE - cg.left.totalE)
	Eslack := slack
	if math.IsInf(Ediff, 0) || math.IsNaN(Ediff) {
		// Overflow protection (mirrors OOMMF's alternate computation).
		Ediff = cg.right.totalE - cg.left.totalE
		Eslack = slack
	}
	fmt.Printf("CUBIC_INPUTS: Ediff=%e lEp=%e rEp=%e span=%e left.Ep=%e right.Ep=%e\n",
		Ediff, lEp, rEp, span, cg.left.Ep, cg.right.Ep)
	// Cubic estimate and its error band.
	cubicPt := 0.5
	cubicErr := 1.0
	if lEp < 0 && (rEp > 0 || Ediff-Eslack >= 0) {
		cubicPt = findCubicMinimum(Ediff, lEp, rEp)
		ca := findCubicMinimum(Ediff+Eslack, lEp, rEp)
		cb := findCubicMinimum(Ediff-Eslack, lEp, rEp)
		if ca > 0 && cb < 1 {
			cubicErr = math.Abs(cb - ca)
		}
	}
	const cubicErrLow = 0.125
	const cubicErrHigh = 0.625

	// Alternative estimate (linear or quadratic on Ep).
	altPt := -1.0
	if cubicErr > cubicErrLow {
		if rEp > 0 {
			dEp := rEp - lEp
			altPt = -lEp / dEp // linear fit on Ep
		} else {
			// Only E-data available; quadratic with limits.
			const rlim = 1.0 / 32
			num := -lEp
			den := 2 * (Ediff - lEp)
			if den <= 0 {
				altPt = 0.5
			} else if num < rlim*den {
				altPt = rlim
			} else if num > (1-rlim)*den {
				altPt = 1 - rlim
			} else {
				altPt = num / den
			}
		}
		altPt = math.Max(0, math.Min(1, altPt))
	}

	// Blend cubic and alternative.
	var lambda float64
	switch {
	case cubicErr <= cubicErrLow:
		lambda = cubicPt
	case cubicErr >= cubicErrHigh:
		lambda = altPt
	default:
		lambda = ((cubicErrHigh-cubicErr)*cubicPt +
			(cubicErr-cubicErrLow)*altPt) /
			(cubicErrHigh - cubicErrLow)
	}

	// Restrict reduction to avoid tiny steps.
	const safety = 1.0 / (1024 * 1024)
	if lambda < 0.25 {
		lambda *= 1 + safety
	} else if lambda > 0.75 {
		lambda *= 1 - safety
	}

	maxRedBase := 0.5
	maxRed := maxRedBase
	if cg.lastMinReductionRatio < maxRed {
		maxRed = cg.lastMinReductionRatio
	}
	if cg.nextToLastMinReductionRatio < maxRed {
		maxRed = cg.nextToLastMinReductionRatio
	}
	maxRed *= maxRed
	if maxRed*span < 1e-15*cg.right.offset {
		t := 1e-15 * cg.right.offset
		if t < 0.5*span {
			maxRed = t / span
		} else {
			maxRed = 0.5
		}
	}
	if span*maxRed < nudge {
		maxRed = nudge / span
		if maxRed > 0.5 {
			maxRed = 0.5
		}
	}
	fmt.Printf("MAXRED_DEBUG: lastRatio=%e nextToLast=%e maxRed_after_min=%.6e maxRed_after_sq=%.6e nudge=%e span=%e right.offset=%e maxRed_final=%e lambda_pre_clamp=%e\n",
		cg.lastMinReductionRatio,
		cg.nextToLastMinReductionRatio,
		func() float64 {
			mr := 0.5
			if cg.lastMinReductionRatio < mr {
				mr = cg.lastMinReductionRatio
			}
			if cg.nextToLastMinReductionRatio < mr {
				mr = cg.nextToLastMinReductionRatio
			}
			return mr
		}(),
		func() float64 {
			mr := 0.5
			if cg.lastMinReductionRatio < mr {
				mr = cg.lastMinReductionRatio
			}
			if cg.nextToLastMinReductionRatio < mr {
				mr = cg.nextToLastMinReductionRatio
			}
			return mr * mr
		}(),
		nudge, span, cg.right.offset, maxRed, lambda)
	if lambda > 0.5 {
		if lambda > 1-maxRed {
			lambda = 1 - maxRed
		}
	} else {
		if lambda < maxRed {
			lambda = maxRed
		}
	}
	fmt.Printf("LAMBDA_DEBUG: cubicPt=%e cubicErr=%e altPt=%e lambda_pre_clamp=%e maxRed=%e lambda_post_clamp=%e\n",
		cubicPt, cubicErr, altPt, lambda, maxRed, lambda)
	testOffset := cg.left.offset + lambda*span
	if testOffset <= cg.left.offset || testOffset >= cg.right.offset {
		lambda = 0.5
		testOffset = 0.5 * (cg.left.offset + cg.right.offset)
		if testOffset <= cg.left.offset || testOffset >= cg.right.offset {
			cg.minFound = true
			cg.lastMinReductionRatio = 0
			cg.nextToLastMinReductionRatio = 0
			return
		}
	}

	cg.lineMinCount++
	cg.fillBracket(testOffset, cg.extra)
	cg.extra = cg.updateBrackets(cg.extra, false)

	if cg.badEdata {
		cg.minFound = true
		cg.bestIsLineMin = false // don't trust this as a real minimum
		cg.lastMinReductionRatio = 0
		cg.nextToLastMinReductionRatio = 0
		return
	}

	newSpan := cg.right.offset - cg.left.offset
	cg.nextToLastMinReductionRatio = cg.lastMinReductionRatio
	cg.lastMinReductionRatio = newSpan / span

	// Check convergence again after update.
	convergenceRHS = cg.bestBracket.gradNorm * cg.basept.dirNorm * cg.anglePrecision * (1 + 2*cg.sumErrEstimate)
	converged = math.Abs(cg.bestBracket.Ep) < convergenceRHS
	fmt.Printf("CONVERGE_CHECK_after update: |bestEp|=%e RHS=%e ratio=%e gradNorm=%e dirNorm=%e dirMaxMagRaw=%e\n",
		math.Abs(cg.bestBracket.Ep),
		cg.bestBracket.gradNorm*cg.basept.dirNorm*cg.anglePrecision*(1+2*cg.sumErrEstimate),
		math.Abs(cg.bestBracket.Ep)/(cg.bestBracket.gradNorm*cg.basept.dirNorm*cg.anglePrecision*(1+2*cg.sumErrEstimate)),
		cg.bestBracket.gradNorm, cg.basept.dirNorm, cg.basept.dirMaxMagRaw)

	fmt.Printf("CONVERGE_GATES_post: converged=%v bestEp_gt_baseptEp=%v bestEp_zero=%v offset_zero=%v span_le_stopSpan=%v nudge_ge_span=%v ratio=%.6e spanReduction=%.6e\n",
		converged,
		cg.bestBracket.Ep > cg.basept.Ep,
		cg.bestBracket.Ep == 0,
		cg.bestBracket.offset == 0,
		span <= cg.stopSpan,
		nudge >= span,
		math.Abs(cg.bestBracket.Ep)/convergenceRHS,
		newSpan/span)
	convergedAndBetter = converged &&
		(cg.bestBracket.Ep == 0 ||
			cg.bestBracket.Ep > cg.basept.Ep ||
			cg.bestBracket.offset == 0)

	if convergedAndBetter &&
		(cg.bestBracket.Ep == 0 ||
			cg.bestBracket.Ep > cg.basept.Ep || // if bestEp > baseptEp, skip span check
			span <= cg.stopSpan ||
			nudge >= span) {
		cg.minFound = true
		cg.bestIsLineMin = true
	} else if cg.right.Ep < 0 {
		// Weak bracket.
		cg.weakBracketCount++
		if cg.weakBracketCount > 4 {
			cg.badEdata = true
			cg.weakBracketCount = 0
			cg.minBracketed = false
		}
	}
	// NOTE: convergenceRHS/converged here are intentionally the values computed
	// above in the "Check convergence again after update" block — nothing
	// affecting them (cg.bestBracket, cg.basept.dirNorm, cg.sumErrEstimate)
	// changes between that block and this print, so recomputing was redundant.
	fmt.Printf("findLineMinimumStep POST-STEP CHECK: newSpan=%e stopSpan=%e nudge=%e bestEp=%e basept.Ep=%e converged=%v rightEp=%e leftEp=%e -> minFound=%v | leftOffset=%.17e rightOffset=%.17e leftOffsetBits=%016x rightOffsetBits=%016x testOffset=%.17e lambda=%e\n",
		newSpan, cg.stopSpan, nudge, cg.bestBracket.Ep, cg.basept.Ep, converged, cg.right.Ep, cg.left.Ep, cg.minFound,
		cg.left.offset, cg.right.offset,
		math.Float64bits(cg.left.offset), math.Float64bits(cg.right.offset),
		testOffset, lambda)
}

// ---------------------------------------------------------------------------
// setBasePoint: establish a new base point and compute the CG direction
// ---------------------------------------------------------------------------

func (cg *CGEvolve) setBasePoint() {
	sz := M.Buffer().Size()
	spin := M.Buffer()

	// DEBUG: check M.Buffer() directly, before any CG processing touches it
	{
		mHost := spin.HostCopy()
		fmt.Printf("setBasePoint RAW M.Buffer(): tilt=(%e,%e,%e) nbr=(%e,%e,%e)\n",
			mHost.Vectors()[0][0][2][2], mHost.Vectors()[1][0][2][2], mHost.Vectors()[2][0][2][2],
			mHost.Vectors()[0][0][2][1], mHost.Vectors()[1][0][2][1], mHost.Vectors()[2][0][2][1])
	}

	if cg.left == nil {
		cg.left = newCGBracket(sz)
		cg.right = newCGBracket(sz)
		cg.extra = newCGBracket(sz)
	}

	cg.cycleCount++
	cg.cycleSubCount++

	nextStepGuess := 0.0
	if cg.bestBracket != nil {
		nextStepGuess = cg.bestBracket.offset
		if cg.bestIsLineMin && cg.left.Ep < 0 && cg.right.Ep > 0 {
			nextStepGuess = (cg.right.Ep*cg.left.offset -
				cg.left.Ep*cg.right.offset) /
				(cg.right.Ep - cg.left.Ep)
		}
	}

	// DEBUG: trace nextStepGuess computation
	fmt.Printf("setBasePoint nextStepGuess: bestIsLineMin=%v bestBracket.offset=%e leftEp=%e rightEp=%e -> nextStepGuess=%e\n",
		cg.bestIsLineMin,
		func() float64 {
			if cg.bestBracket != nil {
				return cg.bestBracket.offset
			}
			return -1
		}(),
		cg.left.Ep, cg.right.Ep, nextStepGuess)
	lastDirNorm := cg.basept.dirNorm

	data.Copy(cg.left.spin, spin)
	cg.left.totalE = cgGetEnergyAndTorque(cg.left.mxHxm)

	cg.left.offset = 0
	cg.left.Eerr = math.Abs(cg.left.totalE) * 8e-16
	bestBefore := cg.bestBracket
	cg.bestBracket = cg.left

	mxHxm := cg.left.mxHxm
	msvRef := cg.msvRef
	msvRef2 := msvRef * msvRef

	if cg.basept.direction == nil {
		cg.basept.direction = cuda.Buffer(3, sz)
	}

	useConjugate := cg.basept.valid &&
		cg.basept.stage == CGStage &&
		cg.cycleSubCount < cg.gradResetCount &&
		cg.basept.direction.Size() == sz

	var (
		gamma     float32 = 0
		newGSumSq float64
	)

	// DEBUG: print both cells' spin going into this cycle
	{
		spinHost := spin.HostCopy()
		fmt.Printf("setBasePoint ENTRY: spin[tilt]=(%e,%e,%e) spin[nbr]=(%e,%e,%e)\n",
			spinHost.Vectors()[0][0][2][2], spinHost.Vectors()[1][0][2][2], spinHost.Vectors()[2][0][2][2],
			spinHost.Vectors()[0][0][2][1], spinHost.Vectors()[1][0][2][1], spinHost.Vectors()[2][0][2][1])
	}

	if useConjugate {
		if cg.basept.method == methodFR {
			newG := float64(cuda.CgGSumSqFR(mxHxm, cg.msv)) * msvRef2
			newGSumSq = newG
			if cg.basept.gSumSq > 0 {
				gamma = float32(newG / cg.basept.gSumSq * cg.basept.dirMaxMagRaw)
			}
		} else { // Polak-Ribière
			if cg.basept.prevMxHxm == nil {
				cg.basept.prevMxHxm = cuda.Buffer(3, sz)
				data.Copy(cg.basept.prevMxHxm, mxHxm)
			}
			res := cuda.CgGammaPR(mxHxm, cg.basept.prevMxHxm, cg.msv)
			newGSumSq = float64(res.GSumSq) * msvRef2
			if cg.basept.gSumSq > 0 {
				gamma = float32(float64(res.GammaSum) * msvRef2 / cg.basept.gSumSq)
			}
			if gamma < 0 {
				gamma = 0
			}
			data.Copy(cg.basept.prevMxHxm, mxHxm)
		}

		{
			dirHost := cg.basept.direction.HostCopy()
			fmt.Printf("[mumax pre-conjugate] gamma=%e dir[tilt]=(%e,%e,%e) dir[nbr]=(%e,%e,%e)\n",
				gamma,
				dirHost.Vectors()[0][0][2][2], dirHost.Vectors()[1][0][2][2], dirHost.Vectors()[2][0][2][2],
				dirHost.Vectors()[0][0][2][1], dirHost.Vectors()[1][0][2][1], dirHost.Vectors()[2][0][2][1])
		}

		res := cuda.CgUpdateDir(cg.basept.direction, mxHxm, spin, cg.msv, gamma)

		fmt.Printf("setBasePoint scaled: Ep=%e normSumSq=%e gradSumSq=%e msvRef=%e msvRef2=%e\n",
			float64(res.Ep)*msvRef, float64(res.NormSumSq)*msvRef2,
			float64(res.GradSumSq)*msvRef2, msvRef, msvRef2)
		fmt.Printf("GRADNORM DEBUG: res.GradSumSq=%e sqrt(GradSumSq)=%e msvRef=%e msvRef2=%e gradNorm=%e\n",
			res.GradSumSq,
			math.Sqrt(float64(res.GradSumSq)),
			msvRef, msvRef2,
			cg.left.gradNorm)

		// DEBUG: per-cell direction + global reduction outputs (main CgUpdateDir call)
		{
			dirHost := cg.basept.direction.HostCopy()
			fmt.Printf("CgUpdateDir[main] RESULT: dir[tilt]=(%e,%e,%e) dir[nbr]=(%e,%e,%e) | MaxMagSq=%e NormSumSq=%e Ep=%e GradSumSq=%e | gamma=%e\n",
				dirHost.Vectors()[0][0][2][2], dirHost.Vectors()[1][0][2][2], dirHost.Vectors()[2][0][2][2],
				dirHost.Vectors()[0][0][2][1], dirHost.Vectors()[1][0][2][1], dirHost.Vectors()[2][0][2][1],
				res.MaxMagSq, res.NormSumSq, res.Ep, res.GradSumSq, gamma)
		}

		dirMaxMagRaw, dirNormRaw = cuda.CgDirNorms(res) // raw kernel-space magnitude
		cg.normalizeDirection(dirMaxMagRaw)             // direction now O(1)
		dirMaxMag := 1.0
		dirNorm := dirNormRaw / dirMaxMagRaw
		cg.basept.dirMaxMagRaw = float64(dirMaxMagRaw)
		Ep := -float64(res.Ep) * msvRef / float64(dirMaxMagRaw)
		cg.left.gradNorm = math.Sqrt(float64(res.GradSumSq)) * msvRef

		if cg.left.gradNorm < float64(gamma)*lastDirNorm*cg.gradResetAngleCotangent {
			cg.gradResetScore = cg.gradResetWgt*cg.gradResetScore +
				(1-cg.gradResetWgt)*1.0
		} else {
			cg.gradResetScore *= cg.gradResetWgt
		}

		if cg.gradResetScore >= cg.gradResetTrigger {
			useConjugate = false
		} else {
			unscaledEp := -float64(res.Ep) * msvRef
			normSumSqScaled := float64(res.NormSumSq) * msvRef2

			epRaw := float64(res.Ep) * msvRef
			rhs := cg.kludgeAdjustAngleCos * math.Sqrt(normSumSqScaled) * cg.left.gradNorm * (1 + 8*1e-15)
			fmt.Printf("[mumax conjugate] res.Ep=%e res.NormSumSq=%e res.GradSumSq=%e\n",
				res.Ep, res.NormSumSq, res.GradSumSq)
			fmt.Printf("[mumax conjugate] epRaw=%e unscaledEp=%e normSumSqScaled=%e gradNorm=%e\n",
				epRaw, unscaledEp, normSumSqScaled, cg.left.gradNorm)
			fmt.Printf("[mumax kludge guard] rhs=%e kludgeFires(epRaw<=rhs)=%v kludgeFires(unscaledEp>=-rhs)=%v\n",
				rhs, epRaw <= rhs, unscaledEp >= -rhs)

			if unscaledEp >= -cg.kludgeAdjustAngleCos*math.Sqrt(normSumSqScaled)*cg.left.gradNorm*(1+8*1e-15) {

				kludge := cg.computeKludgeAlpha(unscaledEp, normSumSqScaled, cg.left.gradNorm)
				fmt.Printf("[mumax kludge] unscaledEp=%e normSumSqScaled=%e gradNorm=%e -> kludgeAlpha=%e\n",
					unscaledEp, normSumSqScaled, cg.left.gradNorm, kludge)
				// NOTE: direction was already normalized above (O(1)).
				// The kludge re-run below recomputes direction from
				// scratch via CgUpdateDir with gamma=1, so it operates
				// on the *normalized* direction as "old dir" input.
				// That's fine: the kludge blends scaledMxHxm into the
				// existing (now O(1)) direction, producing a new,
				// still well-conditioned result, which we normalize
				// again below.
				scaledMxHxm := cuda.Buffer(3, sz)
				defer cuda.Recycle(scaledMxHxm)
				cuda.Madd2(scaledMxHxm, mxHxm, mxHxm, float32(kludge), 0)
				res = cuda.CgUpdateDir(cg.basept.direction,
					scaledMxHxm, spin, cg.msv, 1.0)

				// DEBUG: per-cell direction + global reduction outputs (kludge re-run)
				{
					dirHost := cg.basept.direction.HostCopy()
					fmt.Printf("CgUpdateDir[kludge] RESULT: dir[0]=(%e,%e,%e) dir[1]=(%e,%e,%e) | MaxMagSq=%e NormSumSq=%e Ep=%e GradSumSq=%e | kludge=%e\n",
						dirHost.Vectors()[0][0][0][0], dirHost.Vectors()[1][0][0][0], dirHost.Vectors()[2][0][0][0],
						dirHost.Vectors()[0][0][0][1], dirHost.Vectors()[1][0][0][1], dirHost.Vectors()[2][0][0][1],
						res.MaxMagSq, res.NormSumSq, res.Ep, res.GradSumSq, kludge)
				}

				dirMaxMagRaw, dirNormRaw = cuda.CgDirNorms(res)
				cg.normalizeDirection(dirMaxMagRaw)
				dirMaxMag = 1.0
				dirNorm = dirNormRaw / dirMaxMagRaw

				Ep = -float64(res.Ep) * msvRef / kludge / float64(dirMaxMagRaw)
				cg.left.gradNorm = math.Sqrt(float64(res.GradSumSq)) * msvRef / kludge
			}

			cg.basept.dirMaxMag = dirMaxMag
			cg.basept.dirNorm = dirNorm
			cg.basept.gSumSq = newGSumSq
			cg.left.Ep = Ep
			cg.basept.Ep = Ep
			fmt.Printf("setBasePoint_useconjugate: basept.Ep=%e (unscaled, pre-negation)\n", cg.basept.Ep)

			if Ep < 0 {
				cg.basept.valid = true
			} else {
				useConjugate = false
			}
		}
	}

	if !useConjugate {
		cg.cycleSubCount = 0
		cg.conjCycleCount++
		cg.gradResetScore = 0
		_ = bestBefore

		res := cuda.CgUpdateDir(cg.basept.direction, mxHxm, spin, cg.msv, 0)

		// DEBUG: per-cell direction + global reduction outputs (gradient-reset call)
		{
			dirHost := cg.basept.direction.HostCopy()
			fmt.Printf("CgUpdateDir[reset] RESULT: dir[0]=(%e,%e,%e) dir[1]=(%e,%e,%e) | MaxMagSq=%e NormSumSq=%e Ep=%e GradSumSq=%e\n",
				dirHost.Vectors()[0][0][0][0], dirHost.Vectors()[1][0][0][0], dirHost.Vectors()[2][0][0][0],
				dirHost.Vectors()[0][0][0][1], dirHost.Vectors()[1][0][0][1], dirHost.Vectors()[2][0][0][1],
				res.MaxMagSq, res.NormSumSq, res.Ep, res.GradSumSq)
		}

		dirMaxMagRaw, dirNormRaw = cuda.CgDirNorms(res)
		cg.normalizeDirection(dirMaxMagRaw)
		dirMaxMag := 1.0
		dirNorm := dirNormRaw / dirMaxMagRaw
		cg.basept.dirMaxMagRaw = float64(dirMaxMagRaw)
		Ep := -float64(res.Ep) * msvRef / float64(dirMaxMagRaw)
		cg.left.gradNorm = math.Sqrt(float64(res.GradSumSq)) * msvRef

		cg.basept.dirMaxMag = dirMaxMag
		cg.basept.dirNorm = dirNorm
		cg.basept.gSumSq = float64(cuda.CgGSumSqFR(mxHxm, cg.msv)) * msvRef2
		cg.left.Ep = Ep
		cg.basept.Ep = Ep
		fmt.Printf("setBasePoint_!useconjugate: basept.Ep=%e (unscaled, pre-negation)\n", cg.basept.Ep)
		cg.basept.valid = true
	}

	cg.basept.stage = CGStage
	cg.basept.totalEnergy = cg.left.totalE

	cg.minBracketed = false
	cg.minFound = false
	cg.badEdata = false
	cg.weakBracketCount = 0
	cg.right.offset = -1
	cg.stopSpan = 0

	// direction is now normalized to O(1) magnitude, so offset directly
	// represents the bounded geodesic-step quantity; no division by
	// dirMaxMag is needed (or wanted) here anymore.
	cg.scaledMinstep = cg.minstep
	cg.scaledMaxstep = cg.maxstep

	cg.startStep = cg.scaledMinstep
	if nextStepGuess > 0 {
		cg.startStep = math.Min(nextStepGuess*1.25, cg.scaledMaxstep)
		if cg.startStep < cg.scaledMinstep {
			cg.startStep = cg.scaledMinstep
		}
	}

	cg.lastMinReductionRatio = 1.0 / 16
	cg.nextToLastMinReductionRatio = 1.0 / 256
	cg.bestIsLineMin = false

	// DEBUG: final per-cycle summary, once per setBasePoint call
	{
		dirHost := cg.basept.direction.HostCopy()

		// DEBUG: compute the TRUE post-normalization L2-norm of the direction
		// buffer directly from host data, to compare against the dirNorm
		// value currently being stored in cg.basept.dirNorm.
		// DEBUG: compute the TRUE post-normalization L2-norm AND max-magnitude
		// of the direction buffer directly from host data, to compare against
		// cg.basept.dirNorm and cg.basept.dirMaxMag.
		var trueNormSq float64
		var trueMaxMag float64
		vecs := dirHost.Vectors()
		nx, ny, nz := sz[0], sz[1], sz[2]
		for k := 0; k < nz; k++ {
			for j := 0; j < ny; j++ {
				for i := 0; i < nx; i++ {
					vx := float64(vecs[0][k][j][i])
					vy := float64(vecs[1][k][j][i])
					vz := float64(vecs[2][k][j][i])
					magSq := vx*vx + vy*vy + vz*vz
					trueNormSq += magSq
					if magSq > trueMaxMag {
						trueMaxMag = magSq
					}
				}
			}
		}
		trueNorm := math.Sqrt(trueNormSq)
		trueMaxMag = math.Sqrt(trueMaxMag) // convert from squared

		fmt.Printf(
			"setBasePoint EXIT: dirMaxMagRaw=%e dirNormRaw=%e | dirMaxMag=%e dirNorm=%e | gradNorm=%e | TRUE post-norm L2(direction)=%e TRUE post-norm MAX(direction)=%e | finalDir[tilt]=(%e,%e,%e) finalDir[nbr]=(%e,%e,%e) finalDir[cell512]=(%e,%e,%e) finalDir[cell513]=(%e,%e,%e) finalDir[cell700]=(%e,%e,%e)\n",
			dirMaxMagRaw, dirNormRaw,
			cg.basept.dirMaxMag, cg.basept.dirNorm,
			cg.bestBracket.gradNorm,
			trueNorm, trueMaxMag,

			dirHost.Vectors()[0][0][2][2], dirHost.Vectors()[1][0][2][2], dirHost.Vectors()[2][0][2][2],
			dirHost.Vectors()[0][0][2][1], dirHost.Vectors()[1][0][2][1], dirHost.Vectors()[2][0][2][1],
			dirHost.Vectors()[0][0][8][0], dirHost.Vectors()[1][0][8][0], dirHost.Vectors()[2][0][8][0],
			dirHost.Vectors()[0][0][8][1], dirHost.Vectors()[1][0][8][1], dirHost.Vectors()[2][0][8][1],
			dirHost.Vectors()[0][0][10][60], dirHost.Vectors()[1][0][10][60], dirHost.Vectors()[2][0][10][60],
		)
		fmt.Printf(
			"finalDir[cell512] bits: x=%08x y=%08x z=%08x\n",
			math.Float32bits(dirHost.Vectors()[0][0][8][0]),
			math.Float32bits(dirHost.Vectors()[1][0][8][0]),
			math.Float32bits(dirHost.Vectors()[2][0][8][0]),
		)
	}
}

// computeKludgeAlpha computes the mixing coefficient for KludgeDirection.
// See OOMMF NOTES VII, 16-May-2014, p26-29.
func (cg *CGEvolve) computeKludgeAlpha(rawEp, normsumsq, gradNorm float64) float64 {
	beta2 := cg.kludgeAdjustAngleCos * cg.kludgeAdjustAngleCos * (1 + 8*1e-15)
	N := gradNorm * gradNorm
	A := (1 - beta2) * N * N
	B := 2 * (1 - beta2) * rawEp * N
	C := rawEp*rawEp - beta2*N*normsumsq
	delta := B*B - 4*A*C
	fmt.Printf("[mumax kludge ABC] N=%e beta2=%e A=%e B=%e C=%e delta=%e B*B=%e 4AC=%e\n",
		N, beta2, A, B, C, delta, B*B, 4*A*C)
	const float64Epsilon = 2.220446049250313e-16
	if delta > 0 {
		sqD := math.Sqrt(delta)
		if B >= 0 {
			return -2 * C / (sqD + B)
		}
		return (sqD - B) / (2 * A)
	}
	// Delta <= 0: degenerate case, match OOMMF's fallback exactly
	var alpha float64
	if B > 0 {
		alpha = -2 * C / B
	} else {
		alpha = -B / (2 * A)
	}
	return alpha * (1.0 + 1024*float64Epsilon)
}

// ---------------------------------------------------------------------------
// nudgeBestpt: recovery when line search returns to base point
// ---------------------------------------------------------------------------

func (cg *CGEvolve) nudgeBestpt() {
	nudge := 4 * cgOffsetEpsilon // was: 4 * 1e-15 / cg.basept.dirMaxMag

	testOffset := 0.5
	if cg.right != nil && cg.right.Ep > 0 {
		if -cg.left.Ep < testOffset*(cg.right.Ep-cg.left.Ep) {
			testOffset = -cg.left.Ep / (cg.right.Ep - cg.left.Ep)
		}
	} else {
		// Ep data is not informative; this will get bumped up by
		// the nudge floor check below.
		testOffset = 0.0
	}

	if cg.right != nil {
		testOffset *= cg.right.offset / 2.0
	} else {
		testOffset = 0.0
	}

	// Make sure offset can be felt.
	if testOffset < nudge {
		testOffset = nudge
	}

	cg.lineMinCount++
	cg.fillBracket(testOffset, cg.extra)
	cg.extra = cg.updateBrackets(cg.extra, true)
}

// ---------------------------------------------------------------------------
// Cubic / quadratic interpolation helpers (pure math, no GPU)
// ---------------------------------------------------------------------------

// estimateQuadraticMinimum returns the location of the minimum of the
// least-squares quadratic fit to the four data points (0,f0,fp0,h,fh,fph).
// Returns math.MaxFloat64 if the fit quadratic has no minimum.
// Mirrors OOMMF EstimateQuadraticMinimum.
func estimateQuadraticMinimum(wgt, h, f0, fh, fp0, fph float64) float64 {
	if wgt < 0 || wgt > 1 || h <= 0 {
		return math.MaxFloat64
	}
	fdiff := fh - f0
	fpdiff := fph - fp0
	if fpdiff <= 0 {
		return math.MaxFloat64
	}
	numer := wgt*(0.5*fpdiff-h*fdiff) - 4*(1-wgt)*fp0
	denom := (wgt*h*h + 4*(1-wgt)) * fpdiff
	if denom == 0 {
		return math.MaxFloat64
	}
	return (numer / denom) * h
}

// findCubicMinimum returns the location in [0,1] of the cubic minimum
// given function values scaled to a unit interval.
// Mirrors OOMMF FindCubicMinimum.
func findCubicMinimum(Ediff, lEp, rEp float64) float64 {

	a := -2*Ediff + lEp + rEp
	b := 3*Ediff - 2*lEp - rEp
	c := lEp
	lambda := 0.5

	if a == 0 {
		if b != 0 {
			lambda = -c / (2 * b)
		}
	} else {
		scale := 1.0 / (math.Abs(Ediff) + math.Abs(lEp) + math.Abs(rEp))
		a *= scale
		b *= scale
		c *= scale
		disc := b*b - 3*a*c
		if disc <= 0 {
			disc = 0
		} else {
			disc = math.Sqrt(disc)
		}
		if b >= 0 {
			if math.Abs(c) >= b+disc {
				lambda = sign(-c)
			} else {
				lambda = -c / (b + disc)
			}
		} else {
			if math.Abs(3*a) <= -b+disc {
				lambda = sign(a)
			} else {
				lambda = (-b + disc) / (3 * a)
			}
		}
	}
	fmt.Printf("CUBIC_OUTPUT: Ediff=%e lEp=%e rEp=%e -> lambda=%e\n", Ediff, lEp, rEp, lambda)
	return lambda
}

// ---------------------------------------------------------------------------
// CGMinimize: public entry point (mirrors mumax3's Minimize)
// ---------------------------------------------------------------------------

func CGMinimize() bool {
	CGMinimizerConverged = false
	start := time.Now()

	if CGMinimizerWallClockTime == 0 {
		return false
	}

	SanityCheck()

	// Save and restore solver state.
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time
	relaxing = true
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0
		relaxing = false
	}()

	Precess = false

	if stepper != nil {
		stepper.Free()
	}

	cg := newCGEvolve()
	cg.lastMaxMxHxm = FifoRing(CGMinimizerSamples)
	stepper = cg

	wallOK := func() bool {
		if CGMinimizerWallClockTime < 0 {
			return true
		}
		return time.Since(start) <
			time.Duration(CGMinimizerWallClockTime*float64(time.Second))
	}

	cond := func() bool {
		return (cg.lastMaxMxHxm.count < CGMinimizerSamples ||
			cg.lastMaxMxHxm.Max() > CGMinimizerStop) && wallOK()
	}

	RunWhile(cond)
	pause = true

	CGMinimizerConverged = cg.lastMaxMxHxm.count >= CGMinimizerSamples &&
		cg.lastMaxMxHxm.Max() <= CGMinimizerStop

	stepper.Free()
	return CGMinimizerConverged
}
