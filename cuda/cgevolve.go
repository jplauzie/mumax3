// cuda/cgevolve.go
// Go-side wrappers for the CUDA kernels in cgevolve.cu.
// Follows mumax3's cuda package conventions:
//   - Kernels are launched via k_<name>_async generated wrappers.
//   - Scalar GPU outputs use a 1-element, 1-component Buffer that is
//     copied back to the host with data.Copy after the kernel finishes.
//   - All Slices are [X][Y][Z] component-major (mumax3 layout).

package cuda

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/mumax/3/data"
)

// ---- scalar GPU buffer helpers -------------------------------------------
// mumax3 does not expose a bare float* allocator, so we use a 1-element
// 1-component data.Slice as a single-float GPU cell.

// scalarGPU allocates a 1-element GPU buffer, zeroes it, and returns it.
// The caller must call Recycle on it.
func scalarGPU() *data.Slice {
	b := Buffer(1, [3]int{1, 1, 1})
	Zero(b)
	return b
}

// readScalar copies the single float32 value from a scalarGPU buffer to host.
func readScalar(b *data.Slice) float32 {
	host := data.NewSlice(1, [3]int{1, 1, 1})
	data.Copy(host, b)
	return host.Host()[0][0]
}

// ---- launch configuration ------------------------------------------------

func cgConf(N int) *config {
	return make1DConf(N)
}

// smem returns the shared-memory size (bytes) for a kernel that needs
// nSlots float32 slots per warp.
func smem(blockSize, nSlots int) int {
	nWarps := (blockSize + 31) / 32
	return nSlots * nWarps * int(unsafe.Sizeof(float32(0)))
}

// ---- CgSpinStep ----------------------------------------------------------

// CgSpinStep rotates each spin from the best-point configuration towards the
// search direction by a step.
//
//	dst[i] = normalize( src[i]*sqrt(1 + t0sq*|dir[i]|²) + dvecScale*dir[i] )
//
// src  = spin at best point (offset t0)
// dir  = search direction (basept.direction)
// t0sq = bestpt.offset²
// dvecScale = t_new - t_best
func CgSpinStep(dst, src, dir *data.Slice, t0sq, dvecScale float32) {
	checkSize(dst, src, dir)
	N := dst.Len()
	cfg := cgConf(N)
	k_cgSpinStep_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		src.DevPtr(X), src.DevPtr(Y), src.DevPtr(Z),
		dir.DevPtr(X), dir.DevPtr(Y), dir.DevPtr(Z),
		t0sq, dvecScale, N, cfg)
}

// ---- CgEpAndGradSq -------------------------------------------------------

// CgEpAndGradSq computes the line derivative and gradient norm needed for the
// bracket logic.
//
// Returns:
//
//	ep   = Σ_i (mxHxm[i]·dir[i]) * Ms[i] / sqrt(1 + tsq*|dir[i]|²)
//	gsq  = Σ_i |mxHxm[i]|² * Ms[i]² / (1 + tsq*|dir[i]|²)
//
// Caller converts:
//
//	Ep        = -mu0 * cellVol * ep
//	grad_norm = cellVol * sqrt(gsq)
//
// ms must be a 1-component Slice holding Ms values per cell.
// tsq = current offset².
func CgEpAndGradSq(mxHxm, dir, ms *data.Slice, tsq float32) (ep, gsq float32) {
	N := mxHxm.Len()
	cfg := cgConf(N)

	epBuf := scalarGPU()
	gsqBuf := scalarGPU()
	defer Recycle(epBuf)
	defer Recycle(gsqBuf)

	k_cgEpAndGradSq_async(
		epBuf.DevPtr(0), gsqBuf.DevPtr(0),
		mxHxm.DevPtr(X), mxHxm.DevPtr(Y), mxHxm.DevPtr(Z),
		dir.DevPtr(X), dir.DevPtr(Y), dir.DevPtr(Z),
		ms.DevPtr(0),
		tsq, N, cfg)

	ep = readScalar(epBuf)
	gsq = readScalar(gsqBuf)
	return
}

// ---- CgUpdateDir ---------------------------------------------------------

// CgUpdateDirResult holds the scalar statistics returned by CgUpdateDir.
type CgUpdateDirResult struct {
	MaxMagSq  float32 // max_i |dir[i]|²  (sup-norm²)
	NormSumSq float32 // Σ_i  |dir[i]|²   (L2-norm²)
	Ep        float32 // Σ_i  (dir[i]·mxHxm[i]) * msv[i]  (before -mu0*V)
	GradSumSq float32 // Σ_i  |mxHxm[i]|² * msv[i]²
}

// CgUpdateDir performs the in-place CG direction update:
//
//	dir[i] = msv[i]*mxHxm[i] + gamma*dir[i]
//	dir[i] -= (dir[i]·spin[i]) * spin[i]   // tangent projection
//
// Pass gamma=0 to reset the direction to the preconditioned gradient.
// msv is a 1-component Slice of Ms*cellVolume per cell.
func CgUpdateDir(dir, mxHxm, spin, msv *data.Slice, gamma float32) CgUpdateDirResult {
	N := dir.Len()
	cfg := cgConf(N)

	mmBuf := scalarGPU()
	nsBuf := scalarGPU()
	epBuf := scalarGPU()
	gsBuf := scalarGPU()
	defer Recycle(mmBuf)
	defer Recycle(nsBuf)
	defer Recycle(epBuf)
	defer Recycle(gsBuf)

	k_cgUpdateDir_async(
		dir.DevPtr(X), dir.DevPtr(Y), dir.DevPtr(Z),
		mxHxm.DevPtr(X), mxHxm.DevPtr(Y), mxHxm.DevPtr(Z),
		spin.DevPtr(X), spin.DevPtr(Y), spin.DevPtr(Z),
		msv.DevPtr(0),
		gamma,
		mmBuf.DevPtr(0), nsBuf.DevPtr(0),
		epBuf.DevPtr(0), gsBuf.DevPtr(0),
		N, cfg)
	fmt.Printf("CgUpdateDir: N=%d  cfg=%+v\n", N, cfg)

	raw := CgUpdateDirResult{
		MaxMagSq:  readScalar(mmBuf),
		NormSumSq: readScalar(nsBuf),
		Ep:        readScalar(epBuf),
		GradSumSq: readScalar(gsBuf),
	}
	fmt.Printf("CgUpdateDir RAW results: MaxMagSq=%e NormSumSq=%e Ep=%e GradSumSq=%e gamma=%e\n",
		raw.MaxMagSq, raw.NormSumSq, raw.Ep, raw.GradSumSq, gamma)
	return raw

}

// ---- CgGSumSqFR ----------------------------------------------------------

// CgGSumSqFR returns Σ_i msv[i]² * |mxHxm[i]|².
// Used to compute the Fletcher-Reeves gamma denominator / new numerator.
func CgGSumSqFR(mxHxm, msv *data.Slice) float32 {
	N := mxHxm.Len()
	cfg := cgConf(N)

	outBuf := scalarGPU()
	defer Recycle(outBuf)

	k_cgGSumSqFR_async(
		outBuf.DevPtr(0),
		mxHxm.DevPtr(X), mxHxm.DevPtr(Y), mxHxm.DevPtr(Z),
		msv.DevPtr(0),
		N, cfg)

	return readScalar(outBuf)
}

// ---- CgGammaPR -----------------------------------------------------------

// CgGammaPRResult holds both outputs of the Polak-Ribière accumulation.
type CgGammaPRResult struct {
	GSumSq   float32 // Σ_i msv² |cur|²               (new gradient norm²)
	GammaSum float32 // Σ_i msv² (cur-prev)·cur         (PR numerator)
}

// CgGammaPR computes the Polak-Ribière gamma numerator and new g_sum_sq,
// and updates prevMxHxm in-place to the current mxHxm.
func CgGammaPR(curMxHxm, prevMxHxm, msv *data.Slice) CgGammaPRResult {
	N := curMxHxm.Len()
	cfg := cgConf(N)

	gsBuf := scalarGPU()
	gmBuf := scalarGPU()
	defer Recycle(gsBuf)
	defer Recycle(gmBuf)

	k_cgGammaPR_async(
		gsBuf.DevPtr(0), gmBuf.DevPtr(0),
		curMxHxm.DevPtr(X), curMxHxm.DevPtr(Y), curMxHxm.DevPtr(Z),
		prevMxHxm.DevPtr(X), prevMxHxm.DevPtr(Y), prevMxHxm.DevPtr(Z),
		msv.DevPtr(0),
		N, cfg)

	return CgGammaPRResult{
		GSumSq:   readScalar(gsBuf),
		GammaSum: readScalar(gmBuf),
	}
}

// ---- Utility: build Ms*V slice -------------------------------------------

// CgBuildMsV builds a per-cell weight buffer msv[i] = Ms[i]*cellVolume,
// normalized by a reference scale to avoid denormal float32 values in
// downstream reduction kernels (CgUpdateDir, CgGSumSqFR, CgGammaPR,
// CgEpAndGradSq all multiply or square this buffer).
//
// Returns msvRef such that the *actual* Ms[i]*V = dst[i] * msvRef.
// Callers must un-scale kernel results that were computed using dst:
//   - quantities linear in msv    -> multiply result by msvRef
//   - quantities quadratic in msv -> multiply result by msvRef*msvRef
func CgBuildMsV(dst *data.Slice, ms MSlice, scale float32) (msvRef float32) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_cgBuildMsV_async(dst.DevPtr(0), ms.DevPtr(0), ms.Mul(0), scale, N, cfg)

	msvRef = meanAbsCPU(dst)
	if msvRef == 0 {
		msvRef = 1 // degenerate all-zero Msat case; avoid div-by-zero downstream
	}
	Madd2(dst, dst, dst, 1.0/msvRef, 0) // dst /= msvRef, in place
	return msvRef
}

// meanAbsCPU computes the RMS of a scalar GPU buffer on the CPU, in
// float64, to avoid the same denormal/precision issues we're working
// around on the GPU side. This only runs once per mesh setup (cached
// by ensureMsV's size check), so the host round-trip cost is negligible.
func meanAbsCPU(b *data.Slice) float32 {
	host := b.HostCopy()
	vals := host.Host()[0] // single-component scalar slice
	var sumSq float64
	n := 0
	for _, v := range vals {
		if v == 0 {
			continue // skip non-magnetic cells, matching kernel guards
		}
		sumSq += float64(v) * float64(v)
		n++
	}
	if n == 0 {
		return 0
	}
	return float32(math.Sqrt(sumSq / float64(n)))
}

// CgDirNorms returns (max|dir|, L2|dir|) from the CgUpdateDirResult.
func CgDirNorms(r CgUpdateDirResult) (maxMag, l2Norm float64) {
	return float64(math.Sqrt(float64(r.MaxMagSq))),
		float64(math.Sqrt(float64(r.NormSumSq)))
}
