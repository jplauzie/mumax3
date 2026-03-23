package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func QSin(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qsin_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// cos: dst[i] = cos(src[i])
func QCos(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qcos_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// exp: dst[i] = exp(src[i])
func QExp(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qexp_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// log: dst[i] = log(src[i])
func QLog(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qlog_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// abs: dst[i] = abs(src[i])
func QAbs(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qabs_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// acos: dst[i] = acos(src[i]), returns 0 outside [-1,1]
func QAcos(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qacos_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// acosh: dst[i] = acosh(src[i]), returns 0 for x < 1
func QAcosh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qacosh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// asin: dst[i] = asin(src[i]), returns 0 outside [-1,1]
func QAsin(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qasin_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// asinh: dst[i] = asinh(src[i])
func QAsinh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qasinh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// atan: dst[i] = atan(src[i])
func QAtan(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qatan_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// atan2: dst[i] = atan2(y[i], x[i])
func QAtan2(dst, y, x *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(y.Len() == N && x.Len() == N)
	util.Assert(y.NComp() == nComp && x.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qatan2_async(dst.DevPtr(c), y.DevPtr(c), x.DevPtr(c), N, cfg)
	}
}

// atanh: dst[i] = atanh(src[i]), returns 0 for |x| >= 1
func QAtanh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qatanh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// cosh: dst[i] = cosh(src[i])
func QCosh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qcosh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// erf: dst[i] = erf(src[i])
func QErf(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qerf_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// erfc: dst[i] = erfc(src[i])
func QErfc(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qerfc_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// gamma: dst[i] = gamma(src[i]), returns 0 for x <= 0
func QGamma(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qgamma_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

// heaviside: dst[i] = heaviside(src[i])
// returns 0 for x < 0, 0.5 for x == 0, 1 for x > 0
func QHeaviside(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()

	util.Assert(src.Len() == N && src.NComp() == nComp)

	cfg := make1DConf(N)

	for c := 0; c < nComp; c++ {
		k_Qheaviside_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func QSinc(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qsinc_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func QSinh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qsinh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func QTan(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qtan_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func QTanh(dst, src *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src.Len() == N && src.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qtanh_async(dst.DevPtr(c), src.DevPtr(c), N, cfg)
	}
}

func QMod(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp)
	util.Assert(src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qmod_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}

func QPow(dst, src1, src2 *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src1.NComp() == nComp)
	util.Assert(src2.Len() == N && src2.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_Qpow_async(dst.DevPtr(c), src1.DevPtr(c), src2.DevPtr(c), N, cfg)
	}
}
