package engine

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func init() {
	DeclFunc("Qsin", QSin, "Element-wise sin of a Quantity (scalar or vector)")
	DeclFunc("Qcos", QCos, "Element-wise cos of a Quantity (scalar or vector)")
	DeclFunc("Qtan", QTan, "Element-wise tan of a Quantity (scalar or vector)")
	DeclFunc("Qexp", QExp, "Element-wise exp of a Quantity (scalar or vector)")
	DeclFunc("Qlog", QLog, "Element-wise log of a Quantity (scalar or vector)")
	DeclFunc("Qabs", QAbs, "Element-wise absolute value of a Quantity (scalar or vector)")
	DeclFunc("Qacos", QAcos, "Element-wise arccos of a Quantity (scalar or vector)")
	DeclFunc("Qacosh", QAcosh, "Element-wise arccosh of a Quantity (scalar or vector)")
	DeclFunc("Qasin", QAsin, "Element-wise arcsin of a Quantity (scalar or vector)")
	DeclFunc("Qasinh", QAsinh, "Element-wise arcsinh of a Quantity (scalar or vector)")
	DeclFunc("Qatan", QAtan, "Element-wise arctan of a Quantity (scalar or vector)")
	DeclFunc("Qatan2", QAtan2, "Element-wise atan2(y, x) of two Quantities (scalar or vector)")
	DeclFunc("Qatanh", QAtanh, "Element-wise arctanh of a Quantity (scalar or vector)")
	DeclFunc("Qcosh", QCosh, "Element-wise cosh of a Quantity (scalar or vector)")
	DeclFunc("Qsinh", QSinh, "Element-wise sinh of a Quantity (scalar or vector)")
	DeclFunc("Qtanh", QTanh, "Element-wise tanh of a Quantity (scalar or vector)")
	DeclFunc("Qerf", QErf, "Element-wise error function of a Quantity (scalar or vector)")
	DeclFunc("Qerfc", QErfc, "Element-wise complementary error function of a Quantity (scalar or vector)")
	DeclFunc("Qgamma", QGamma, "Element-wise gamma function of a Quantity (scalar or vector)")
	DeclFunc("Qheaviside", QHeaviside, "Element-wise Heaviside step function of a Quantity (scalar or vector)")
	DeclFunc("Qsinc", QSinc, "Element-wise sinc function of a Quantity (scalar or vector)")
	DeclFunc("Qpow", QPow, "Element-wise power: Qpow(base, exp)")
	DeclFunc("Qmod", QMod, "Element-wise modulo: Qmod(a, b)")
}

type unaryFuncQuantity struct {
	a    Quantity
	f    func(dst, a *data.Slice)
	name string
}

func (q *unaryFuncQuantity) NComp() int { return q.a.NComp() }

func (q *unaryFuncQuantity) EvalTo(dst *data.Slice) {
	a := ValueOf(q.a)
	defer cuda.Recycle(a)
	q.f(dst, a)
}

func newUnaryQ(a Quantity, f func(dst, a *data.Slice), name string) Quantity {
	return &unaryFuncQuantity{a, f, name}
}

//pow, mod and atan2 are binary functions, so they need a separate struct and constructor

type binaryFuncQuantity struct {
	a, b  Quantity
	nComp int
	f     func(dst, a, b *data.Slice)
	name  string
}

func (q *binaryFuncQuantity) NComp() int { return q.nComp }

func (q *binaryFuncQuantity) EvalTo(dst *data.Slice) {
	a := ValueOf(q.a)
	defer cuda.Recycle(a)
	b := ValueOf(q.b)
	defer cuda.Recycle(b)

	switch {
	case a.NComp() == b.NComp():
		q.f(dst, a, b)
	case a.NComp() == 1:
		// broadcast a across each component
		for c := 0; c < b.NComp(); c++ {
			q.f(dst.Comp(c), a, b.Comp(c))
		}
	case b.NComp() == 1:
		// broadcast b across each component
		for c := 0; c < a.NComp(); c++ {
			q.f(dst.Comp(c), a.Comp(c), b)
		}
	}
}

func newBinaryQ(a, b Quantity, f func(dst, a, b *data.Slice), name string) Quantity {
	nComp := -1
	switch {
	case a.NComp() == b.NComp():
		nComp = a.NComp()
	case a.NComp() == 1:
		nComp = b.NComp()
	case b.NComp() == 1:
		nComp = a.NComp()
	default:
		panic(fmt.Sprintf("Cannot apply %v to %v and %v components", name, a.NComp(), b.NComp()))
	}
	return &binaryFuncQuantity{a, b, nComp, f, name}
}

func QSin(a Quantity) Quantity       { return newUnaryQ(a, cuda.QSin, "Qsin") }
func QCos(a Quantity) Quantity       { return newUnaryQ(a, cuda.QCos, "Qcos") }
func QTan(a Quantity) Quantity       { return newUnaryQ(a, cuda.QTan, "Qtan") }
func QExp(a Quantity) Quantity       { return newUnaryQ(a, cuda.QExp, "Qexp") }
func QLog(a Quantity) Quantity       { return newUnaryQ(a, cuda.QLog, "Qlog") }
func QAbs(a Quantity) Quantity       { return newUnaryQ(a, cuda.QAbs, "Qabs") }
func QAcos(a Quantity) Quantity      { return newUnaryQ(a, cuda.QAcos, "Qacos") }
func QAcosh(a Quantity) Quantity     { return newUnaryQ(a, cuda.QAcosh, "Qacosh") }
func QAsin(a Quantity) Quantity      { return newUnaryQ(a, cuda.QAsin, "Qasin") }
func QAsinh(a Quantity) Quantity     { return newUnaryQ(a, cuda.QAsinh, "Qasinh") }
func QAtan(a Quantity) Quantity      { return newUnaryQ(a, cuda.QAtan, "Qatan") }
func QAtanh(a Quantity) Quantity     { return newUnaryQ(a, cuda.QAtanh, "Qatanh") }
func QCosh(a Quantity) Quantity      { return newUnaryQ(a, cuda.QCosh, "Qcosh") }
func QSinh(a Quantity) Quantity      { return newUnaryQ(a, cuda.QSinh, "Qsinh") }
func QTanh(a Quantity) Quantity      { return newUnaryQ(a, cuda.QTanh, "Qtanh") }
func QErf(a Quantity) Quantity       { return newUnaryQ(a, cuda.QErf, "Qerf") }
func QErfc(a Quantity) Quantity      { return newUnaryQ(a, cuda.QErfc, "Qerfc") }
func QGamma(a Quantity) Quantity     { return newUnaryQ(a, cuda.QGamma, "Qgamma") }
func QHeaviside(a Quantity) Quantity { return newUnaryQ(a, cuda.QHeaviside, "Qheaviside") }
func QSinc(a Quantity) Quantity      { return newUnaryQ(a, cuda.QSinc, "Qsinc") }
func QPow(a, b Quantity) Quantity    { return newBinaryQ(a, b, cuda.QPow, "Qpow") }
func QMod(a, b Quantity) Quantity    { return newBinaryQ(a, b, cuda.QMod, "Qmod") }
func QAtan2(y, x Quantity) Quantity  { return newBinaryQ(y, x, cuda.QAtan2, "Qatan2") }
