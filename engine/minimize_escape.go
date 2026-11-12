package engine

// Local-minimum certification and escape attempts for Minimize().
//
// MinimizerEscape = 0: off (default)
//                 = 1: after convergence, run Lanczos on the Riemannian Hessian and report the
//                      lowest eigenvalues (EscapeLambdaMin). The state is not changed.
//                 = 2: additionally, kick the state along the softest Ritz modes (both signs,
//                      several amplitudes), relax each kicked state with the TR minimizer, and
//                      accept the first result whose energy is lower than the start state.
//                      Repeats (new Lanczos at the new state) up to EscapeMaxRounds times.
//                      (Straight-line kicks cannot follow a curved path, e.g. a vortex core
//                      moving tens of nm; kept for comparison.)
//                 = 3: minimum-mode following. Starting from the softest modes, repeatedly
//                      (a) push the state a small step along the current lowest-curvature mode,
//                      (b) relax all directions orthogonal to that mode with the TR minimizer
//                          (hyperplane-constrained), (c) update the mode with a warm-started
//                      Lanczos. The path therefore follows the curved soft valley uphill. When
//                      the force along the mode changes sign the ridge has been crossed; the
//                      state is then relaxed freely and accepted if its energy is lower.
//                      Both directions of each soft mode are tried.
//                 = 4: basin hopping with spatially correlated noise. Perturb the state with a
//                      smooth random rotation field (
//                      correlation length and amplitude sampled log-uniformly), relax with TR, and
//                      accept the first result with lower energy. Cheap per trial (one
//                      relaxation, no Hessian analysis), and the perturbation can change the
//                      state at the scale of domains/vortices, so it can make large topological
//                      moves that following a single soft mode cannot afford.
//
// Eigenvalues are those of the Hessian of E/(Ms*V) with respect to per-cell rotation, i.e. in
// the same units as the torque (Tesla). Negative = unstable direction (saddle / unfinished).
// Like the TR minimizer this assumes uniform Msat.
//
// Note: the relaxations here always use the TR minimizer, whatever MinimizerTR says.
// Frozen spins appear as exact zero eigenvalues.

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	MinimizerEscape   int     = 0    // 0 off, 1 Lanczos report, 2 report + kicks, 3 report + minimum-mode following
	EscapeLanczosIter int     = 150  // max Lanczos iterations (basis is kept in host memory)
	EscapeLanczosTol  float64 = 0.05 // Ritz residual tolerance, relative to max(|lambda|, gap to neighbouring Ritz value)
	EscapeModes       int     = 3    // number of soft modes to kick along
	EscapeMaxAmp      float64 = 3.0  // largest kick: max per-cell |v| of the retraction vector (~rad for small values)
	EscapeLevels      int     = 4    // amplitudes MaxAmp, MaxAmp/2, ... (EscapeLevels of them)
	EscapeMaxRelax    int     = 12   // max number of relaxations per round
	EscapeMaxRounds   int     = 50   // max number of accepted escapes
	EscapeTol         float64 = 1e-4 // a state counts as lower if E < E0 - EscapeTol*|E0|
	EscapeSeed        int     = 1    // seed for the random Lanczos start vector
	EscapeVerbose     bool    = true // print progress to stdout

	EscapeStep          float64 = 0.1 // mode following: push per step, as max per-cell rotation vector length
	EscapeFollowSteps   int     = 300 // mode following: max push steps per direction
	EscapeRelaxSteps    int     = 20  // mode following: max constrained TR steps after each push
	EscapeHopTrials     int     = 24  // basin hopping: perturb-and-relax trials per round
	EscapeHopAmp        float64 = 1.0 // basin hopping: base amplitude (max per-cell rotation vector length); trials sample 0.5x .. 4x
	EscapeHopLength     float64 = 16  // basin hopping: base correlation length of the random field (cells); trials sample 1/4x .. 4x
	EscapeHopPatience   int     = 25  // basin hopping: stop when this many consecutive trials return to the start state (0 = never)
	EscapeMaxCross      int     = 5   // mode following: max candidate crossings tested per direction
	EscapeLogEvery      int     = 5   // print every n-th push step (mode 3) / hop trial (mode 4); crossings and accepted results always
	EscapeRelaxTol      float64 = 0.1 // mode following: stop relaxing when |transverse torque| <= tol*|force along mode|
	EscapeFollowLanczos int     = 30  // mode following: max Lanczos iterations per mode update (warm start)
	EscapeMaxRise       float64 = 0.5 // mode following: give up a direction if E rises above E0 by this fraction of |E0|

	EscapeLambdaMin float64 // lowest Hessian eigenvalue found in the last Lanczos run
	EscapeLanczosOK int     // 1 if the last Lanczos run met its residual tolerance, else 0
	EscapeEStart    float64 // total energy when the escape analysis started (after the plain TR relaxation)
	EscapeEFinal    float64 // total energy when the escape analysis finished
	EscapeAccepted  int     // number of escapes accepted in the last call
)

func init() {
	DeclVar("MinimizerEscape", &MinimizerEscape, "After Minimize converges: 0 = off, 1 = Lanczos report of lowest Hessian eigenvalues, 2 = report + try to escape by kicking along soft modes and relaxing.")
	DeclVar("EscapeLanczosIter", &EscapeLanczosIter, "Max Lanczos iterations for the escape analysis (memory: that many vectors).")
	DeclVar("EscapeLanczosTol", &EscapeLanczosTol, "Lanczos Ritz-residual tolerance relative to the largest Ritz value.")
	DeclVar("EscapeModes", &EscapeModes, "Number of soft modes used for kicks.")
	DeclVar("EscapeMaxAmp", &EscapeMaxAmp, "Largest kick amplitude (max per-cell rotation vector length).")
	DeclVar("EscapeLevels", &EscapeLevels, "Number of kick amplitudes (halving from EscapeMaxAmp).")
	DeclVar("EscapeMaxRelax", &EscapeMaxRelax, "Max relaxations per escape round.")
	DeclVar("EscapeMaxRounds", &EscapeMaxRounds, "Max accepted escapes per Minimize call.")
	DeclVar("EscapeTol", &EscapeTol, "Relative energy decrease needed to accept an escape.")
	DeclVar("EscapeSeed", &EscapeSeed, "Random seed for the Lanczos start vector.")
	DeclVar("EscapeVerbose", &EscapeVerbose, "Print escape progress to stdout.")
	DeclVar("EscapeStep", &EscapeStep, "Mode following: push per step (max per-cell rotation vector length).")
	DeclVar("EscapeFollowSteps", &EscapeFollowSteps, "Mode following: max push steps per direction.")
	DeclVar("EscapeRelaxSteps", &EscapeRelaxSteps, "Mode following: max constrained TR steps after each push.")
	DeclVar("EscapeHopTrials", &EscapeHopTrials, "Basin hopping: perturb-and-relax trials per round.")
	DeclVar("EscapeHopAmp", &EscapeHopAmp, "Basin hopping: base perturbation amplitude (max per-cell rotation vector length).")
	DeclVar("EscapeHopLength", &EscapeHopLength, "Basin hopping: base correlation length of the random rotation field in cells (trials sample 1/4x to 4x).")
	DeclVar("EscapeHopPatience", &EscapeHopPatience, "Basin hopping: stop after this many consecutive trials that return to the start state (0 = never).")
	DeclVar("EscapeMaxCross", &EscapeMaxCross, "Mode following: max candidate ridge crossings tested per direction.")
	DeclVar("EscapeLogEvery", &EscapeLogEvery, "Print every n-th push step (mode 3) or hop trial (mode 4); accepted results are always printed.")
	DeclVar("EscapeRelaxTol", &EscapeRelaxTol, "Mode following: stop transverse relaxation when |transverse torque| <= tol*|force along mode|.")
	DeclVar("EscapeFollowLanczos", &EscapeFollowLanczos, "Mode following: max Lanczos iterations per mode update.")
	DeclVar("EscapeMaxRise", &EscapeMaxRise, "Mode following: abort a direction if E exceeds E0 by this fraction of |E0|.")
	DeclVar("EscapeLambdaMin", &EscapeLambdaMin, "Lowest Hessian eigenvalue (Tesla) found by the last escape analysis.")
	DeclVar("EscapeLanczosOK", &EscapeLanczosOK, "1 if the last Lanczos run converged to its tolerance, else 0.")
	DeclVar("EscapeEStart", &EscapeEStart, "Total energy (J) at the start of the last escape analysis.")
	DeclVar("EscapeEFinal", &EscapeEFinal, "Total energy (J) at the end of the last escape analysis.")
	DeclVar("EscapeAccepted", &EscapeAccepted, "Number of escapes accepted in the last Minimize call.")
}

func escLog(format string, a ...interface{}) {
	if EscapeVerbose {
		fmt.Printf("[escape] "+format+"\n", a...)
	}
}

// Entry point, called by Minimize() after convergence.
func escape(timerStart time.Time) {
	EscapeAccepted = 0
	escLog("escape analysis v10 (result variables for scripts)")
	m := M.Buffer()
	best := cuda.Buffer(3, m.Size())
	defer cuda.Recycle(best)

	for round := 0; round < EscapeMaxRounds || MinimizerEscape < 2 && round == 0; round++ {
		if !WallclockTimer(timerStart, MinimizeWallClockTime) {
			break
		}
		data.Copy(best, m)
		e0 := GetTotalEnergy()
		if round == 0 {
			EscapeEStart = e0
		}

		if MinimizerEscape == 4 {
			escLog("round %d: E = %g, basin hopping", round, e0)
			if !tryHop(best, e0, int64(EscapeSeed+round), timerStart) {
				break
			}
			EscapeAccepted++
			continue
		}

		res := lanczosLowest(EscapeLanczosIter, EscapeModes, EscapeModes, EscapeLanczosTol, int64(EscapeSeed+round), nil, nil)
		EscapeLambdaMin = res.theta[0]
		status := "converged"
		EscapeLanczosOK = 1
		if !res.converged {
			EscapeLanczosOK = 0
			status = "NOT converged, eigenvalues are only estimates"
		}
		rel := make([]string, len(res.theta))
		for k := range res.theta {
			rel[k] = fmt.Sprintf("%.1f%%", 100*res.resid[k]/math.Max(math.Abs(res.theta[k]), 1e-30))
		}
		escLog("round %d: E = %g, Lanczos %d its (%s), lowest eigenvalues %v, relative residuals %v",
			round, e0, res.iters, status, res.theta, rel)
		if MinimizerEscape < 2 {
			res.free()
			break
		}

		var improved bool
		if MinimizerEscape == 3 {
			improved = tryFollow(best, e0, &res, timerStart)
		} else {
			improved = tryKicks(best, e0, &res, timerStart)
		}
		res.free()
		if !improved {
			break
		}
		EscapeAccepted++
	}

	EscapeEFinal = GetTotalEnergy()
	escLog("finished: %d accepted escape(s), final E = %g", EscapeAccepted, EscapeEFinal)

	// report torque of the state we end up in
	t := cuda.Buffer(3, m.Size())
	torqueFn(t)
	setMaxTorque(t)
	cuda.Recycle(t)
}

// relax from the current magnetization with the TR minimizer
func relaxTR(timerStart time.Time) {
	tr := &trMinimizer{lastDm: FifoRing(DmSamples)}
	stepper = tr
	cond := func() bool {
		return (tr.lastDm.count < DmSamples || tr.lastDm.Max() > StopMaxDm) &&
			WallclockTimer(timerStart, MinimizeWallClockTime)
	}
	RunWhile(cond)
	pause = true
	tr.Free()
}

// Kicks the state `best` along the Ritz modes and relaxes. Returns true if M then holds a
// relaxed state with lower energy than e0. Otherwise M is restored to `best`.
func tryKicks(best *data.Slice, e0 float64, res *lanczosResult, timerStart time.Time) bool {
	m := M.Buffer()
	tol := EscapeTol * math.Abs(e0)

	// scale modes so that the largest per-cell component has length 1
	for _, v := range res.vec {
		if vmax := cuda.MaxVecNorm(v); vmax > 0 {
			cuda.Madd2(v, v, v, float32(1/vmax), 0)
		}
	}

	type cand struct {
		mode  int
		amp   float64 // signed
		probe float64 // energy of the kicked, unrelaxed state
	}
	var cands []cand
	for k, v := range res.vec {
		for _, sgn := range []float64{1, -1} {
			for l := 0; l < EscapeLevels; l++ {
				amp := sgn * EscapeMaxAmp / math.Pow(2, float64(l))
				cuda.Minimize(m, best, v, float32(amp))
				M.normalize()
				cands = append(cands, cand{k, amp, GetTotalEnergy()})
			}
		}
	}
	data.Copy(m, best)

	// candidates whose unrelaxed energy is already lower first, then largest kicks first
	sort.SliceStable(cands, func(i, j int) bool {
		pi, pj := cands[i].probe < e0-tol, cands[j].probe < e0-tol
		if pi != pj {
			return pi
		}
		if pi {
			return cands[i].probe < cands[j].probe
		}
		return math.Abs(cands[i].amp) > math.Abs(cands[j].amp)
	})

	for n, c := range cands {
		if n >= EscapeMaxRelax || !WallclockTimer(timerStart, MinimizeWallClockTime) {
			break
		}
		cuda.Minimize(m, best, res.vec[c.mode], float32(c.amp))
		M.normalize()
		relaxTR(timerStart)
		e := GetTotalEnergy()
		escLog("  kick mode %d amp %+.3g (probe E %g) -> relaxed E %g", c.mode, c.amp, c.probe, e)
		if e < e0-tol {
			escLog("  accepted: E %g -> %g", e0, e)
			return true
		}
	}
	data.Copy(m, best)
	return false
}

// Minimum-mode following from the state `best` (energy e0) along each of the soft Ritz modes of
// `res`, in both directions. Returns true if M then holds a relaxed state with energy < e0.
// Otherwise M is restored to `best`.
func tryFollow(best *data.Slice, e0 float64, res *lanczosResult, timerStart time.Time) bool {
	m := M.Buffer()
	size := m.Size()
	tol := EscapeTol * math.Abs(e0)

	v := cuda.Buffer(3, size)    // current soft mode (unit 2-norm), also the TR constraint
	base := cuda.Buffer(3, size) // base point for retractions
	defer cuda.Recycle(v)
	defer cuda.Recycle(base)

	for k := 0; k < len(res.vec) && k < EscapeModes; k++ {
		for _, sgn := range []float64{1, -1} {
			data.Copy(m, best)
			if !WallclockTimer(timerStart, MinimizeWallClockTime) {
				return false
			}
			cuda.Madd2(v, res.vec[k], res.vec[k], float32(sgn), 0)
			escLog("  follow mode %d (lambda %+.3e), direction %+.0f", k, res.theta[k], sgn)
			if followMode(best, base, v, e0, tol, timerStart) {
				return true
			}
		}
	}
	data.Copy(m, best)
	return false
}

// v: unit tangent vector at the current M (modified in place); base: scratch buffer.
//
// A detected crossing (force along the mode changes sign) is only a candidate: the mode can
// change character near the edge of the valley, so the state is tested by pushing 1, 3 and 10
// steps beyond it and relaxing freely. If none of these ends below e0, the crossing was a false
// alarm and the following resumes from the crossing state.
func followMode(best, base, v *data.Slice, e0, tol float64, timerStart time.Time) bool {
	m := M.Buffer()
	size := m.Size()

	newTR := func() *trMinimizer { return &trMinimizer{lastDm: FifoRing(DmSamples), avoid: v} }
	tr := newTR()
	defer func() { tr.Free() }()

	cross := cuda.Buffer(3, size) // state at a detected crossing
	defer cuda.Recycle(cross)

	nrelax := EscapeRelaxSteps
	if nrelax < 1 {
		nrelax = 1
	}
	scale := 1.0 // adaptive step scale: shrinks where the mode turns unstable (near the ridge)
	push := func() bool {
		vmax := cuda.MaxVecNorm(v)
		if vmax == 0 {
			return false
		}
		data.Copy(base, m)
		cuda.Minimize(m, base, v, float32(EscapeStep*scale/vmax))
		M.normalize()
		return true
	}

	ncross, skip := 0, 0
	var seen []float64 // relaxed energies of earlier failed crossing tests
	for it := 0; it < EscapeFollowSteps && WallclockTimer(timerStart, MinimizeWallClockTime); it++ {
		// (a) push along the mode
		if !push() {
			break
		}
		// (b) relax all directions orthogonal to the mode
		tr.refreshTorque()
		nr, ratio := 0, 0.0
		for i := 0; i < nrelax; i++ {
			tr.Step()
			nr = i + 1
			tt, vt := trDot(tr.t, tr.t), trDot(v, tr.t)
			ratio = math.Sqrt(math.Max(0, tt-vt*vt)) / math.Max(math.Abs(vt), 1e-30)
			if ratio <= EscapeRelaxTol {
				break
			}
		}
		// (c) update the mode at the new point, keeping its orientation
		// borrow the TR minimizer's idle scratch vectors as Lanczos workspace (saves GPU buffers)
		lr := lanczosLowest(EscapeFollowLanczos, 1, 1, EscapeLanczosTol, 0, v,
			[]*data.Slice{tr.s, tr.r, tr.d, tr.hd, tr.tmp})
		if len(lr.vec) == 0 {
			break
		}
		if trDot(lr.vec[0], v) < 0 {
			cuda.Madd2(lr.vec[0], lr.vec[0], lr.vec[0], -1, 0)
		}
		data.Copy(v, lr.vec[0])
		lambda := lr.theta[0]
		lr.free()

		e := GetTotalEnergy()
		f := trDot(tr.t, v) // force along +v: negative while climbing, positive once past the ridge
		if EscapeLogEvery <= 1 || it%EscapeLogEvery == 0 || f > 0 {
			cuda.Madd2(cross, m, best, 1, -1)
			rms := math.Sqrt(trDot(cross, cross) / float64(m.Len()))
			escLog("    step %3d: E-E0 = %+.3e, lambda = %+.3e, force along mode = %+.3e (%d relax steps, transverse/force %.2f, step scale %.2f, rms dm from start %.4f)",
				it, e-e0, lambda, f, nr, ratio, scale, rms)
		}
		if lambda < 0 {
			scale = math.Max(scale*0.5, 1.0/16)
		} else {
			scale = math.Min(scale*1.25, 1)
		}
		if skip > 0 {
			skip--
		}

		if f > 0 && skip == 0 && ratio <= 5*EscapeRelaxTol { // a sign change is only meaningful if the other directions are relaxed
			// candidate crossing: test it with free relaxations from beyond the crossing
			ncross++
			data.Copy(cross, m)
			tr.Free() // release the GPU scratch of the constrained minimizer first
			vmax := cuda.MaxVecNorm(v)
			if vmax > 0 {
				for _, mult := range []float64{1, 3, 10} {
					cuda.Minimize(m, cross, v, float32(mult*EscapeStep/vmax))
					M.normalize()
					relaxTR(timerStart)
					er := GetTotalEnergy()
					cuda.Madd2(base, m, best, 1, -1)
					escLog("  crossing %d (E-E0 = %+.3e): %g steps beyond -> relaxed E = %g (dE/|E0| = %+.2e), rms dm from start %.4f (max %.2f)",
						ncross, e-e0, mult, er, (er-e0)/math.Abs(e0),
						math.Sqrt(trDot(base, base)/float64(m.Len())), cuda.MaxVecNorm(base))
					if er < e0-tol {
						escLog("  accepted: E %g -> %g", e0, er)
						return true
					}
					if mult == 1 {
						for _, prev := range seen {
							if math.Abs(er-prev) <= 1e-5*math.Abs(e0) {
								escLog("    this crossing leads to the same state as an earlier one, giving up this direction")
								data.Copy(m, best)
								return false
							}
						}
						seen = append(seen, er)
					}
				}
			}
			// false alarm: resume following from the crossing state
			data.Copy(m, cross)
			tr = newTR()
			skip = 3
			if ncross >= EscapeMaxCross {
				escLog("    %d crossings tested without finding a lower state, giving up this direction", ncross)
				break
			}
			continue
		}
		if e-e0 > EscapeMaxRise*math.Abs(e0) {
			escLog("    energy rose too far, giving up this direction")
			break
		}
	}
	data.Copy(m, best)
	return false
}

// Basin hopping from the state `best` (energy e0): returns true if M then holds a relaxed state
// with lower energy; otherwise M is restored to `best`.
func tryHop(best *data.Slice, e0 float64, seed int64, timerStart time.Time) bool {
	m := M.Buffer()
	size := m.Size()
	tol := EscapeTol * math.Abs(e0)
	rng := rand.New(rand.NewSource(seed))

	u := cuda.Buffer(3, size)
	defer cuda.Recycle(u)

	same := 0 // consecutive trials that ended in the start state
	for t := 0; t < EscapeHopTrials; t++ {
		if !WallclockTimer(timerStart, MinimizeWallClockTime) {
			break
		}
		// sample the correlation length (1/4x..4x of the base) and amplitude (0.5x..4x) log-uniformly
		length := EscapeHopLength * math.Pow(4, 2*rng.Float64()-1)
		amp := EscapeHopAmp * math.Pow(2, 3*rng.Float64()-1)

		randomSmoothTangent(u, rng, length) // M == best here
		cuda.Minimize(m, best, u, float32(amp))
		M.normalize()
		relaxTR(timerStart)
		e := GetTotalEnergy()
		rel := (e - e0) / math.Abs(e0)
		if EscapeLogEvery <= 1 || t%EscapeLogEvery == 0 || e < e0-tol {
			escLog("  hop %2d (length %5.1f cells, amp %.2f): relaxed E = %g (dE/|E0| = %+.2e)", t, length, amp, e, rel)
		}
		if e < e0-tol {
			escLog("  accepted: E %g -> %g", e0, e)
			return true
		}
		data.Copy(m, best)
		if math.Abs(rel) <= 1e-5 {
			same++
		} else {
			same = 0
		}
		if EscapeHopPatience > 0 && same >= EscapeHopPatience {
			escLog("  %d consecutive trials returned to the start state (after %d trials): stopping", same, t+1)
			return false
		}
	}
	return false
}

// Spatially correlated random tangent field: Gaussian random vectors on a coarse lattice with
// spacing `length` cells (random offset), smoothstep-interpolated onto the grid, projected onto
// the tangent space of the current M, zero outside the geometry, scaled so that the largest
// per-cell vector has length 1. Cost is O(cells) for any correlation length.
func randomSmoothTangent(dst *data.Slice, rng *rand.Rand, length float64) {
	mh := M.Buffer().HostCopy()
	mv := mh.Host()
	size := mh.Size()
	N := mh.Len()
	if length < 1 {
		length = 1
	}

	var g [3]int // coarse lattice nodes per axis
	var off [3]float64
	for a := 0; a < 3; a++ {
		g[a] = 1
		if size[a] > 1 {
			g[a] = int(math.Ceil(float64(size[a])/length)) + 2
			off[a] = rng.Float64()
		}
	}
	node := make([][3]float64, g[0]*g[1]*g[2])
	for i := range node {
		for c := 0; c < 3; c++ {
			node[i][c] = rng.NormFloat64()
		}
	}

	rv := data.NewSlice(3, size)
	r := rv.Host()
	for z := 0; z < size[2]; z++ {
		for y := 0; y < size[1]; y++ {
			for x := 0; x < size[0]; x++ {
				pos := [3]int{x, y, z}
				var i0 [3]int
				var w [3][2]float64
				for a := 0; a < 3; a++ {
					if g[a] == 1 {
						w[a] = [2]float64{1, 0}
						continue
					}
					f := (float64(pos[a])+0.5)/length + off[a]
					fl := math.Floor(f)
					i0[a] = int(fl)
					t := f - fl
					sm := t * t * (3 - 2*t)
					w[a] = [2]float64{1 - sm, sm}
				}
				var v [3]float64
				for dz := 0; dz < 2; dz++ {
					for dy := 0; dy < 2; dy++ {
						for dx := 0; dx < 2; dx++ {
							wt := w[0][dx] * w[1][dy] * w[2][dz]
							if wt == 0 {
								continue
							}
							k := (i0[0] + dx) + g[0]*((i0[1]+dy)+g[1]*(i0[2]+dz))
							for c := 0; c < 3; c++ {
								v[c] += wt * node[k][c]
							}
						}
					}
				}
				i := x + size[0]*(y+size[1]*z)
				r[0][i], r[1][i], r[2][i] = float32(v[0]), float32(v[1]), float32(v[2])
			}
		}
	}

	maxn := float32(0)
	for i := 0; i < N; i++ {
		mx, my, mz := mv[0][i], mv[1][i], mv[2][i]
		if mx*mx+my*my+mz*mz < 0.25 { // outside the geometry
			r[0][i], r[1][i], r[2][i] = 0, 0, 0
			continue
		}
		d := mx*r[0][i] + my*r[1][i] + mz*r[2][i]
		r[0][i] -= d * mx
		r[1][i] -= d * my
		r[2][i] -= d * mz
		n := r[0][i]*r[0][i] + r[1][i]*r[1][i] + r[2][i]*r[2][i]
		if n > maxn {
			maxn = n
		}
	}
	if maxn > 0 {
		f := 1 / float32(math.Sqrt(float64(maxn)))
		for c := 0; c < 3; c++ {
			for i := 0; i < N; i++ {
				r[c][i] *= f
			}
		}
	}
	data.Copy(dst, rv)
}

type lanczosResult struct {
	theta     []float64     // lowest Ritz values, ascending
	resid     []float64     // Ritz residual norms
	vec       []*data.Slice // Ritz vectors, unit 2-norm (caller frees with free())
	iters     int
	converged bool // the first nConv Ritz pairs met the residual tolerance
}

func (r *lanczosResult) free() {
	for _, v := range r.vec {
		cuda.Recycle(v)
	}
	r.vec = nil
}

// random vector tangent to m, zero outside the geometry (|m| < 0.5)
func randomTangent(dst *data.Slice, seed int64) {
	mh := M.Buffer().HostCopy()
	mv := mh.Host()
	N := mh.Len()
	rv := data.NewSlice(3, mh.Size())
	rh := rv.Host()
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < N; i++ {
		mx, my, mz := mv[0][i], mv[1][i], mv[2][i]
		if mx*mx+my*my+mz*mz < 0.25 {
			continue
		}
		x, y, z := float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())
		d := mx*x + my*y + mz*z
		rh[0][i], rh[1][i], rh[2][i] = x-d*mx, y-d*my, z-d*mz
	}
	data.Copy(dst, rv)
}

// host-side vector helpers for the Lanczos basis (host slices are not subject to the
// cuda.Buffer count limit, which is ~100 vector components in total)
func hostDot(a, b *data.Slice) float64 {
	x, y := a.Host(), b.Host()
	s := 0.0
	for c := range x {
		for i := range x[c] {
			s += float64(x[c][i]) * float64(y[c][i])
		}
	}
	return s
}

// y += f*x
func hostAxpy(y, x *data.Slice, f float64) {
	yy, xx := y.Host(), x.Host()
	for c := range yy {
		for i := range yy[c] {
			yy[c][i] += float32(f * float64(xx[c][i]))
		}
	}
}

func hostScale(y *data.Slice, f float64) {
	yy := y.Host()
	for c := range yy {
		for i := range yy[c] {
			yy[c][i] = float32(f * float64(yy[c][i]))
		}
	}
}

// Lanczos with full reorthogonalization on the (finite-difference) Riemannian Hessian at the
// current magnetization. Returns Ritz vectors for the nVec lowest eigenvalues; iteration stops when the lowest
// nConv of them have residual <= tol*max(|theta|, gap to the neighbouring Ritz value).
// If start != nil (device vector) it is projected onto the tangent space and used as the start
// vector (warm start), otherwise a random tangent vector is used.
// work, if non-nil, supplies 5 scratch GPU vectors (borrowed, not freed) so that no extra
// cuda.Buffers are allocated; otherwise they are allocated and released here.
// The Krylov basis lives in host memory; only the current vector and the Hessian-vector product
// live on the GPU (3 + 3 + workspace buffers), at the cost of two host<->device copies per
// iteration and O(iter^2) host-side dot products.
func lanczosLowest(maxIter, nVec, nConv int, tol float64, seed int64, start *data.Slice, work []*data.Slice) lanczosResult {
	m := M.Buffer()
	size := m.Size()

	// GPU workspace: m0, tmp, tmp2 for hessVecCentral, plus the current vector and H*q
	var ws *trMinimizer
	var qdev, wdev *data.Slice
	if work != nil {
		ws = &trMinimizer{m0: work[0], tmp: work[1], tmp2: work[2]}
		qdev, wdev = work[3], work[4]
	} else {
		ws = &trMinimizer{}
		ws.m0 = cuda.Buffer(3, size)
		ws.tmp = cuda.Buffer(3, size)
		ws.tmp2 = cuda.Buffer(3, size)
		qdev = cuda.Buffer(3, size)
		wdev = cuda.Buffer(3, size)
		defer ws.Free()
		defer cuda.Recycle(qdev)
		defer cuda.Recycle(wdev)
	}
	data.Copy(ws.m0, m)

	var Q []*data.Slice // host basis
	q0 := data.NewSlice(3, size)
	if start == nil {
		randomTangent(q0, seed)
	} else {
		data.Copy(qdev, data.NewSlice(3, size)) // host zeros (qdev is overwritten later)
		cuda.TRHessVec(wdev, start, qdev, m, 1) // wdev = -P_m start
		cuda.Madd2(wdev, wdev, wdev, -1, 0)
		data.Copy(q0, wdev)
	}
	nrm := math.Sqrt(hostDot(q0, q0))
	if nrm == 0 { // nothing to analyze
		return lanczosResult{theta: []float64{0}, resid: []float64{0}}
	}
	hostScale(q0, 1/nrm)
	Q = append(Q, q0)
	w := data.NewSlice(3, size) // host

	var alpha, beta []float64
	var theta []float64
	var vecs [][]float64
	var lastBeta float64
	converged := false
	n := 0
	for j := 0; j < maxIter; j++ {
		data.Copy(qdev, Q[j])
		ws.hessVecCentral(wdev, qdev)
		data.Copy(w, wdev)

		a := hostDot(w, Q[j])
		alpha = append(alpha, a)
		hostAxpy(w, Q[j], -a)
		if j > 0 {
			hostAxpy(w, Q[j-1], -beta[j-1])
		}
		for pass := 0; pass < 2; pass++ { // full reorthogonalization, twice is enough
			for i := 0; i <= j; i++ {
				hostAxpy(w, Q[i], -hostDot(w, Q[i]))
			}
		}
		b := math.Sqrt(hostDot(w, w))
		n = j + 1
		lastBeta = b

		interval := 10
		if n <= 100 {
			interval = 5
		}
		if n <= 40 {
			interval = 2
		}
		done := b < 1e-9 || n == maxIter
		if done || (n >= nConv+2 && n%interval == 0) {
			theta, vecs = tridiagEig(alpha, beta)
			ok := true
			for k := 0; k < nConv && k < n; k++ {
				ref := math.Abs(theta[k])
				gap := math.Inf(1)
				if k > 0 {
					gap = math.Min(gap, theta[k]-theta[k-1])
				}
				if k+1 < n {
					gap = math.Min(gap, theta[k+1]-theta[k])
				}
				if !math.IsInf(gap, 1) && gap > ref {
					ref = gap
				}
				if b*math.Abs(vecs[k][n-1]) > tol*ref {
					ok = false
				}
			}
			converged = ok || b < 1e-9
			done = done || ok
		}
		if done {
			break
		}
		beta = append(beta, b)
		q := data.NewSlice(3, size)
		hostAxpy(q, w, 1/b)
		Q = append(Q, q)
	}

	res := lanczosResult{iters: n, converged: converged}
	yh := data.NewSlice(3, size)
	for k := 0; k < nVec && k < n; k++ {
		yy := yh.Host()
		for c := range yy {
			for i := range yy[c] {
				yy[c][i] = 0
			}
		}
		for i := 0; i < n; i++ {
			hostAxpy(yh, Q[i], vecs[k][i])
		}
		if nn := math.Sqrt(hostDot(yh, yh)); nn > 0 {
			hostScale(yh, 1/nn)
		}
		y := cuda.Buffer(3, size) // Ritz vector on the GPU, owned by the result
		data.Copy(y, yh)
		res.vec = append(res.vec, y)
		res.theta = append(res.theta, theta[k])
		res.resid = append(res.resid, lastBeta*math.Abs(vecs[k][n-1]))
	}
	return res
}

// eigen-decomposition of the symmetric tridiagonal matrix with diagonal alpha (n) and
// off-diagonal beta (n-1). vals ascending, vecs[k][i] = component i of eigenvector k.
func tridiagEig(alpha, beta []float64) (vals []float64, vecs [][]float64) {
	n := len(alpha)
	a := make([][]float64, n)
	for i := range a {
		a[i] = make([]float64, n)
		a[i][i] = alpha[i]
	}
	for i := 0; i < n-1; i++ {
		a[i][i+1] = beta[i]
		a[i+1][i] = beta[i]
	}
	return symEig(a)
}

// cyclic Jacobi eigenvalue algorithm for a small dense symmetric matrix (destroys a).
func symEig(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	v := make([][]float64, n)
	for i := range v {
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for sweep := 0; sweep < 100; sweep++ {
		off, tot := 0.0, 0.0
		for p := 0; p < n; p++ {
			for q := 0; q < n; q++ {
				x := a[p][q] * a[p][q]
				tot += x
				if p != q {
					off += x
				}
			}
		}
		if off <= 1e-26*tot {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p][q]
				if apq == 0 {
					continue
				}
				theta := (a[q][q] - a[p][p]) / (2 * apq)
				var t float64
				if math.Abs(theta) > 1e100 {
					t = 1 / (2 * theta)
				} else {
					t = 1 / (math.Abs(theta) + math.Sqrt(theta*theta+1))
					if theta < 0 {
						t = -t
					}
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := 0; k < n; k++ {
					akp, akq := a[k][p], a[k][q]
					a[k][p], a[k][q] = c*akp-s*akq, s*akp+c*akq
				}
				for k := 0; k < n; k++ {
					apk, aqk := a[p][k], a[q][k]
					a[p][k], a[q][k] = c*apk-s*aqk, s*apk+c*aqk
				}
				for k := 0; k < n; k++ {
					vkp, vkq := v[k][p], v[k][q]
					v[k][p], v[k][q] = c*vkp-s*vkq, s*vkp+c*vkq
				}
			}
		}
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool { return a[idx[i]][idx[i]] < a[idx[j]][idx[j]] })
	vals = make([]float64, n)
	vecs = make([][]float64, n)
	for k, id := range idx {
		vals[k] = a[id][id]
		vecs[k] = make([]float64, n)
		for i := 0; i < n; i++ {
			vecs[k][i] = v[i][id]
		}
	}
	return vals, vecs
}
