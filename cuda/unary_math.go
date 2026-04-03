package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// applyUnary is a helper that applies a per-component async kernel
// to every component of dst and a. dst and a must have the same shape.
func applyUnary(dst, a *data.Slice, kernel func(dst, a unsafe.Pointer, N int, cfg *config)) {
	N := dst.Len()
	cfg := make1DConf(N)
	for c := 0; c < dst.NComp(); c++ {
		kernel(dst.DevPtr(c), a.DevPtr(c), N, cfg)
	}
}

// applyUnaryScalar is a helper for kernels that take an additional scalar parameter.
func applyUnaryScalar(dst, a *data.Slice, b float32, kernel func(dst, a unsafe.Pointer, b float32, N int, cfg *config)) {
	N := dst.Len()
	cfg := make1DConf(N)
	for c := 0; c < dst.NComp(); c++ {
		kernel(dst.DevPtr(c), a.DevPtr(c), b, N, cfg)
	}
}

// QSin computes dst[i] = sin(a[i]) for each component.
func QSin(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sin_async) }

// QCos computes dst[i] = cos(a[i]) for each component.
func QCos(dst, a *data.Slice) { applyUnary(dst, a, k_unary_cos_async) }

// QTan computes dst[i] = tan(a[i]) for each component.
func QTan(dst, a *data.Slice) { applyUnary(dst, a, k_unary_tan_async) }

// QExp computes dst[i] = exp(a[i]) for each component.
func QExp(dst, a *data.Slice) { applyUnary(dst, a, k_unary_exp_async) }

// QLog computes dst[i] = log(a[i]) for each component, 0 for non-positive inputs.
func QLog(dst, a *data.Slice) { applyUnary(dst, a, k_unary_log_async) }

// QAbs computes dst[i] = |a[i]| for each component.
func QAbs(dst, a *data.Slice) { applyUnary(dst, a, k_unary_abs_async) }

// QAcos computes dst[i] = acos(a[i]) for each component.
func QAcos(dst, a *data.Slice) { applyUnary(dst, a, k_unary_acos_async) }

// QAcosh computes dst[i] = acosh(a[i]) for each component.
func QAcosh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_acosh_async) }

// QAsin computes dst[i] = asin(a[i]) for each component.
func QAsin(dst, a *data.Slice) { applyUnary(dst, a, k_unary_asin_async) }

// QAsinh computes dst[i] = asinh(a[i]) for each component.
func QAsinh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_asinh_async) }

// QAtan computes dst[i] = atan(a[i]) for each component.
func QAtan(dst, a *data.Slice) { applyUnary(dst, a, k_unary_atan_async) }

// QAtanh computes dst[i] = atanh(a[i]) for each component.
func QAtanh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_atanh_async) }

// QCosh computes dst[i] = cosh(a[i]) for each component.
func QCosh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_cosh_async) }

// QSinh computes dst[i] = sinh(a[i]) for each component.
func QSinh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sinh_async) }

// QTanh computes dst[i] = tanh(a[i]) for each component.
func QTanh(dst, a *data.Slice) { applyUnary(dst, a, k_unary_tanh_async) }

// QErf computes dst[i] = erf(a[i]) for each component.
func QErf(dst, a *data.Slice) { applyUnary(dst, a, k_unary_erf_async) }

// QErfc computes dst[i] = erfc(a[i]) for each component.
func QErfc(dst, a *data.Slice) { applyUnary(dst, a, k_unary_erfc_async) }

// QGamma computes dst[i] = tgamma(a[i]) for each component.
func QGamma(dst, a *data.Slice) { applyUnary(dst, a, k_unary_gamma_async) }

// QHeaviside computes dst[i] = 0 if a[i] < 0, 0.5 if a[i] == 0, 1 if a[i] > 0.
func QHeaviside(dst, a *data.Slice) { applyUnary(dst, a, k_unary_heaviside_async) }

// QSinc computes dst[i] = sin(a[i])/a[i] for each component, 1 for a[i] == 0.
func QSinc(dst, a *data.Slice) { applyUnary(dst, a, k_unary_sinc_async) }

func QPow(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp)
	util.Assert(src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_pow_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}

func QMod(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp)
	util.Assert(src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_mod_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}

func QAtan2(dst, y, x *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(y.Len() == N && y.NComp() == nComp)
	util.Assert(x.Len() == N && x.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_pw_atan2_async(dst.DevPtr(c), y.DevPtr(c), x.DevPtr(c), N, cfg)
	}
}
