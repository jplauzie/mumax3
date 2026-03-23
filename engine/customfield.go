package engine

// Add arbitrary terms to B_eff, Edens_total.

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	B_custom       = NewVectorField("B_custom", "T", "User-defined field", AddCustomField)
	Edens_custom   = NewScalarField("Edens_custom", "J/m3", "Energy density of user-defined field.", AddCustomEnergyDensity)
	E_custom       = NewScalarValue("E_custom", "J", "total energy of user-defined field", GetCustomEnergy)
	customTerms    []Quantity // vector
	customEnergies []Quantity // scalar
)

func init() {
	registerEnergy(GetCustomEnergy, AddCustomEnergyDensity)
	DeclFunc("AddFieldTerm", AddFieldTerm, "Add an expression to B_eff.")
	DeclFunc("AddEdensTerm", AddEdensTerm, "Add an expression to Edens.")
	DeclFunc("Add", Add, "Add two quantities")
	DeclFunc("Madd", Madd, "Weighted addition: Madd(Q1,Q2,c1,c2) = c1*Q1 + c2*Q2")
	DeclFunc("Dot", Dot, "Dot product of two vector quantities")
	DeclFunc("Cross", Cross, "Cross product of two vector quantities")
	DeclFunc("Mul", Mul, "Point-wise product of two quantities")
	DeclFunc("MulMV", MulMV, "Matrix-Vector product: MulMV(AX, AY, AZ, m) = (AX·m, AY·m, AZ·m). "+
		"The arguments Ax, Ay, Az and m are quantities with 3 componets.")
	DeclFunc("Div", Div, "Point-wise division of two quantities")
	DeclFunc("Const", Const, "Constant, uniform number")
	DeclFunc("ConstVector", ConstVector, "Constant, uniform vector")
	DeclFunc("Shifted", Shifted, "Shifted quantity")
	DeclFunc("Masked", Masked, "Mask quantity with shape")
	DeclFunc("Normalized", Normalized, "Normalize quantity")
	DeclFunc("RemoveCustomFields", RemoveCustomFields, "Removes all custom fields again")
	DeclFunc("RemoveCustomEnergies", RemoveCustomEnergies, "Removes all custom energies")
	DeclFunc("RunningAverage", RunningAverage, "Records the time-average of a quantity from the moment this function is called.<br>Note: this may impact performance since the Quantity will be evaluated after every step.")
	DeclFunc("Sum", Sum, "Sum of Quantity over all cells in the grid. For a vector Quantity, all components are added together.")
	DeclFunc("SumVector", SumVector, "Sum of vector Quantity over all cells in the grid.")
	DeclFunc("QSin", QSin, "Pointwise sine of a quantity: QSin(q)")
	DeclFunc("QCos", QCos, "Pointwise cosine of a quantity: QCos(q)")
	DeclFunc("QExp", QExp, "Pointwise exponential of a quantity: QExp(q)")
	DeclFunc("QLog", QLog, "Pointwise natural logarithm of a quantity: QLog(q)")
	DeclFunc("QAbs", QAbs, "Pointwise absolute value of a quantity: QAbs(q)")
	DeclFunc("QAcos", QAcos, "Pointwise arccos of a quantity: QAcos(q)")
	DeclFunc("QAcosh", QAcosh, "Pointwise inverse hyperbolic cosine: QAcosh(q)")
	DeclFunc("QAsin", QAsin, "Pointwise arcsine of a quantity: QAsin(q)")
	DeclFunc("QAsinh", QAsinh, "Pointwise inverse hyperbolic sine: QAsinh(q)")
	DeclFunc("QAtan", QAtan, "Pointwise arctangent of a quantity: QAtan(q)")
	DeclFunc("QAtanh", QAtanh, "Pointwise inverse hyperbolic tangent: QAtanh(q)")
	DeclFunc("QCosh", QCosh, "Pointwise hyperbolic cosine: QCosh(q)")
	DeclFunc("QSinh", QSinh, "Pointwise hyperbolic sine: QSinh(q)")
	DeclFunc("QTan", QTan, "Pointwise tangent: QTan(q)")
	DeclFunc("QTanh", QTanh, "Pointwise hyperbolic tangent: QTanh(q)")
	DeclFunc("QErf", QErf, "Pointwise error function: QErf(q)")
	DeclFunc("QErfc", QErfc, "Pointwise complementary error function: QErfc(q)")
	DeclFunc("QGamma", QGamma, "Pointwise gamma function: QGamma(q)")
	DeclFunc("QHeaviside", QHeaviside, "Pointwise Heaviside step function: QHeaviside(q)")
	DeclFunc("QSinc", QSinc, "Pointwise normalized sinc function: QSinc(q)")
	DeclFunc("QMod", QMod, "Pointwise modulo: QMod(a,b)")
	DeclFunc("QPow", QPow, "Pointwise power: QPow(a,b)")
}

// Removes all customfields
func RemoveCustomFields() {
	customTerms = nil
}

// Removes all customenergies
func RemoveCustomEnergies() {
	customEnergies = nil
}

// AddFieldTerm adds an effective field function (returning Teslas) to B_eff.
// Be sure to also add the corresponding energy term using AddEnergyTerm.
func AddFieldTerm(b Quantity) {
	customTerms = append(customTerms, b)
}

// AddEnergyTerm adds an energy density function (returning Joules/m³) to Edens_total.
// Needed when AddFieldTerm was used and a correct energy is needed
// (e.g. for Relax, Minimize, ...).
func AddEdensTerm(e Quantity) {
	customEnergies = append(customEnergies, e)
}

// AddCustomField evaluates the user-defined custom field terms
// and adds the result to dst.
func AddCustomField(dst *data.Slice) {
	for _, term := range customTerms {
		buf := ValueOf(term)
		cuda.Add(dst, dst, buf)
		cuda.Recycle(buf)
	}
}

// Adds the custom energy densities (defined with AddEdensTerm)
func AddCustomEnergyDensity(dst *data.Slice) {
	for _, term := range customEnergies {
		buf := ValueOf(term)
		cuda.Add(dst, dst, buf)
		cuda.Recycle(buf)
	}
}

func GetCustomEnergy() float64 {
	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)
	cuda.Zero(buf)
	AddCustomEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}

type constValue struct {
	value []float64
}

func (c *constValue) NComp() int { return len(c.value) }

func (d *constValue) EvalTo(dst *data.Slice) {
	for c, v := range d.value {
		cuda.Memset(dst.Comp(c), float32(v))
	}
}

// Const returns a constant (uniform) scalar quantity,
// that can be used to construct custom field terms.
func Const(v float64) Quantity {
	return &constValue{[]float64{v}}
}

// ConstVector returns a constant (uniform) vector quantity,
// that can be used to construct custom field terms.
func ConstVector(x, y, z float64) Quantity {
	return &constValue{[]float64{x, y, z}}
}

// fieldOp holds the abstract functionality for operations
// (like add, multiply, ...) on space-dependent quantites
// (like M, B_sat, ...)
type fieldOp struct {
	a, b  Quantity
	nComp int
}

func (o fieldOp) NComp() int {
	return o.nComp
}

type dotProduct struct {
	fieldOp
}

type crossProduct struct {
	fieldOp
}

type addition struct {
	fieldOp
}

type mAddition struct {
	fieldOp
	fac1, fac2 float64
}

type mulmv struct {
	ax, ay, az, b Quantity
}

// MulMV returns a new Quantity that evaluates to the
// matrix-vector product (Ax·b, Ay·b, Az·b).
func MulMV(Ax, Ay, Az, b Quantity) Quantity {
	util.Argument(Ax.NComp() == 3 &&
		Ay.NComp() == 3 &&
		Az.NComp() == 3 &&
		b.NComp() == 3)
	return &mulmv{Ax, Ay, Az, b}
}

func (q *mulmv) EvalTo(dst *data.Slice) {
	util.Argument(dst.NComp() == 3)
	cuda.Zero(dst)
	b := ValueOf(q.b)
	defer cuda.Recycle(b)

	{
		Ax := ValueOf(q.ax)
		cuda.AddDotProduct(dst.Comp(X), 1, Ax, b)
		cuda.Recycle(Ax)
	}
	{

		Ay := ValueOf(q.ay)
		cuda.AddDotProduct(dst.Comp(Y), 1, Ay, b)
		cuda.Recycle(Ay)
	}
	{
		Az := ValueOf(q.az)
		cuda.AddDotProduct(dst.Comp(Z), 1, Az, b)
		cuda.Recycle(Az)
	}
}

func (q *mulmv) NComp() int {
	return 3
}

// DotProduct creates a new quantity that is the dot product of
// quantities a and b. E.g.:
//
//	DotProct(&M, &B_ext)
func Dot(a, b Quantity) Quantity {
	return &dotProduct{fieldOp{a, b, 1}}
}

func (d *dotProduct) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.AddDotProduct(dst, 1, A, B)
}

// CrossProduct creates a new quantity that is the cross product of
// quantities a and b. E.g.:
//
//	CrossProct(&M, &B_ext)
func Cross(a, b Quantity) Quantity {
	return &crossProduct{fieldOp{a, b, 3}}
}

func (d *crossProduct) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.CrossProduct(dst, A, B)
}

func Add(a, b Quantity) Quantity {
	if a.NComp() != b.NComp() {
		panic(fmt.Sprintf("Cannot point-wise Add %v components by %v components", a.NComp(), b.NComp()))
	}
	return &addition{fieldOp{a, b, a.NComp()}}
}

func (d *addition) EvalTo(dst *data.Slice) {
	A := ValueOf(d.a)
	defer cuda.Recycle(A)
	B := ValueOf(d.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.Add(dst, A, B)
}

type pointwiseMul struct {
	fieldOp
}

func Madd(a, b Quantity, fac1, fac2 float64) *mAddition {
	if a.NComp() != b.NComp() {
		panic(fmt.Sprintf("Cannot point-wise add %v components by %v components", a.NComp(), b.NComp()))
	}
	return &mAddition{fieldOp{a, b, a.NComp()}, fac1, fac2}
}

func (o *mAddition) EvalTo(dst *data.Slice) {
	A := ValueOf(o.a)
	defer cuda.Recycle(A)
	B := ValueOf(o.b)
	defer cuda.Recycle(B)
	cuda.Zero(dst)
	cuda.Madd2(dst, A, B, float32(o.fac1), float32(o.fac2))
}

// Mul returns a new quantity that evaluates to the pointwise product a and b.
func Mul(a, b Quantity) Quantity {
	nComp := -1
	switch {
	case a.NComp() == b.NComp():
		nComp = a.NComp() // vector*vector, scalar*scalar
	case a.NComp() == 1:
		nComp = b.NComp() // scalar*something
	case b.NComp() == 1:
		nComp = a.NComp() // something*scalar
	default:
		panic(fmt.Sprintf("Cannot point-wise multiply %v components by %v components", a.NComp(), b.NComp()))
	}

	return &pointwiseMul{fieldOp{a, b, nComp}}
}

func (d *pointwiseMul) EvalTo(dst *data.Slice) {
	cuda.Zero(dst)
	a := ValueOf(d.a)
	defer cuda.Recycle(a)
	b := ValueOf(d.b)
	defer cuda.Recycle(b)

	switch {
	case a.NComp() == b.NComp():
		mulNN(dst, a, b) // vector*vector, scalar*scalar
	case a.NComp() == 1:
		mul1N(dst, a, b)
	case b.NComp() == 1:
		mul1N(dst, b, a)
	default:
		panic(fmt.Sprintf("Cannot point-wise multiply %v components by %v components", a.NComp(), b.NComp()))
	}
}

// mulNN pointwise multiplies two N-component vectors,
// yielding an N-component vector stored in dst.
func mulNN(dst, a, b *data.Slice) {
	cuda.Mul(dst, a, b)
}

// mul1N pointwise multiplies a scalar (1-component) with an N-component vector,
// yielding an N-component vector stored in dst.
func mul1N(dst, a, b *data.Slice) {
	util.Assert(a.NComp() == 1)
	util.Assert(dst.NComp() == b.NComp())
	for c := 0; c < dst.NComp(); c++ {
		cuda.Mul(dst.Comp(c), a, b.Comp(c))
	}
}

type pointwiseDiv struct {
	fieldOp
}

// Div returns a new quantity that evaluates to the pointwise product a and b.
func Div(a, b Quantity) Quantity {
	nComp := -1
	switch {
	case a.NComp() == b.NComp():
		nComp = a.NComp() // vector/vector, scalar/scalar
	case b.NComp() == 1:
		nComp = a.NComp() // something/scalar
	default:
		panic(fmt.Sprintf("Cannot point-wise divide %v components by %v components", a.NComp(), b.NComp()))
	}
	return &pointwiseDiv{fieldOp{a, b, nComp}}
}

func (d *pointwiseDiv) EvalTo(dst *data.Slice) {
	a := ValueOf(d.a)
	defer cuda.Recycle(a)
	b := ValueOf(d.b)
	defer cuda.Recycle(b)

	switch {
	case a.NComp() == b.NComp():
		divNN(dst, a, b) // vector*vector, scalar*scalar
	case b.NComp() == 1:
		divN1(dst, a, b)
	default:
		panic(fmt.Sprintf("Cannot point-wise divide %v components by %v components", a.NComp(), b.NComp()))
	}

}

func divNN(dst, a, b *data.Slice) {
	cuda.Div(dst, a, b)
}

func divN1(dst, a, b *data.Slice) {
	util.Assert(dst.NComp() == a.NComp())
	util.Assert(b.NComp() == 1)
	for c := 0; c < dst.NComp(); c++ {
		cuda.Div(dst.Comp(c), a.Comp(c), b)
	}
}

type shifted struct {
	orig       Quantity
	dx, dy, dz int
}

// Shifted returns a new Quantity that evaluates to
// the original, shifted over dx, dy, dz cells.
func Shifted(q Quantity, dx, dy, dz int) Quantity {
	util.Assert(dx != 0 || dy != 0 || dz != 0)
	return &shifted{q, dx, dy, dz}
}

func (q *shifted) EvalTo(dst *data.Slice) {
	orig := ValueOf(q.orig)
	defer cuda.Recycle(orig)
	for i := 0; i < q.NComp(); i++ {
		dsti := dst.Comp(i)
		origi := orig.Comp(i)
		if q.dx != 0 {
			cuda.ShiftX(dsti, origi, q.dx, 0, 0)
			data.Copy(origi, dsti)
		}
		if q.dy != 0 {
			cuda.ShiftY(dsti, origi, q.dy, 0, 0)
			data.Copy(origi, dsti)
		}
		if q.dz != 0 {
			cuda.ShiftZ(dsti, origi, q.dz, 0, 0)
		}
	}
}

func (q *shifted) NComp() int {
	return q.orig.NComp()
}

// Masks a quantity with a shape
// The shape will only be evaluated once on the mesh,
// and will be re-evaluated after mesh change,
// because otherwise too slow
func Masked(q Quantity, shape Shape) Quantity {
	return &masked{q, shape, nil, data.Mesh{}}
}

type masked struct {
	orig  Quantity
	shape Shape
	mask  *data.Slice
	mesh  data.Mesh
}

func (q *masked) EvalTo(dst *data.Slice) {
	if q.mesh != *Mesh() {
		// When mesh is changed, mask needs an update
		q.createMask()
	}
	orig := ValueOf(q.orig)
	defer cuda.Recycle(orig)
	mul1N(dst, q.mask, orig)
}

func (q *masked) NComp() int {
	return q.orig.NComp()
}

func (q *masked) createMask() {
	size := Mesh().Size()
	// Prepare mask on host
	maskhost := data.NewSlice(SCALAR, size)
	defer maskhost.Free()
	maskScalars := maskhost.Scalars()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				if q.shape(r[X], r[Y], r[Z]) {
					maskScalars[iz][iy][ix] = 1
				}
			}
		}
	}
	// Update mask
	q.mask.Free()
	q.mask = cuda.NewSlice(SCALAR, size)
	data.Copy(q.mask, maskhost)
	q.mesh = *Mesh()
	// Remove mask from host
}

// Normalized returns a quantity that evaluates to the unit vector of q
func Normalized(q Quantity) Quantity {
	return &normalized{q}
}

type normalized struct {
	orig Quantity
}

func (q *normalized) NComp() int {
	return 3
}

func (q *normalized) EvalTo(dst *data.Slice) {
	util.Assert(dst.NComp() == q.NComp())
	q.orig.EvalTo(dst)
	cuda.Normalize(dst, nil)
}

// RunningAverage returns the running average of a quantity
// over time, starting at the moment RunningAverage() is called.
// This value is updated after every Step() and depends on the time step.
func RunningAverage(q Quantity) Quantity {
	ra := runningAverage{q, nil, Time, 0}
	ra.avg = cuda.Buffer(q.NComp(), SizeOf(q))
	cuda.Zero(ra.avg)
	PostStep(func() {
		dt := Time - ra.prev_t
		if dt < 0 { // Don't update the time average if we went back in time since the last step
			return
		}
		ra.prev_t = Time
		ra.total_t += dt
		val := ValueOf(q)
		defer cuda.Recycle(val)
		cuda.Madd2(ra.avg, ra.avg, val, float32((ra.total_t-dt)/ra.total_t), float32(dt/ra.total_t))
	})
	return &ra
}

type runningAverage struct {
	orig    Quantity
	avg     *data.Slice
	prev_t  float64
	total_t float64
}

func (ra *runningAverage) EvalTo(dst *data.Slice) {
	util.Assert(dst.NComp() == ra.NComp())
	data.Copy(dst, ra.avg)
}

func (ra *runningAverage) NComp() int {
	return ra.orig.NComp()
}

// Sum of Quantity over all cells in the grid.
// For a vector Quantity, all components are added together.
func Sum(q Quantity) float64 {
	val := ValueOf(q)
	defer cuda.Recycle(val)
	total := 0.
	for i := 0; i < q.NComp(); i++ {
		total += float64(cuda.Sum(val.Comp(i)))
	}
	return total
}

// Sum of vector Quantity over all cells in the grid.
func SumVector(q Quantity) data.Vector {
	util.Assert(q.NComp() == 3)
	val := ValueOf(q)
	defer cuda.Recycle(val)
	var v [3]float64
	for i := 0; i < 3; i++ {
		v[i] = float64(cuda.Sum(val.Comp(i)))
	}
	return Vector(v[0], v[1], v[2])
}

var TimeQ = NewScalarField("TimeQ", "s", "Simulation time", func(dst *data.Slice) {
	cuda.Memset(dst, float32(Time))
})

var TimeQVec = NewVectorField("TimeQVec", "s", "Simulation time (vector)", func(dst *data.Slice) {
	t := float32(Time)
	for c := 0; c < 3; c++ {
		cuda.Memset(dst.Comp(c), t)
	}
})

// QSin returns a Quantity representing sin(q) evaluated pointwise
func QSin(q Quantity) Quantity {
	return &sinOp{q}
}

type sinOp struct {
	orig Quantity
}

func (s *sinOp) NComp() int {
	return s.orig.NComp()
}

func (s *sinOp) EvalTo(dst *data.Slice) {
	src := ValueOf(s.orig)
	defer cuda.Recycle(src)
	cuda.QSin(dst, src)
}

// QCos returns a Quantity representing cos(q) evaluated pointwise
func QCos(q Quantity) Quantity {
	return &cosOp{q}
}

type cosOp struct{ orig Quantity }

func (op *cosOp) NComp() int { return op.orig.NComp() }
func (op *cosOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QCos(dst, src)
}

// QExp: pointwise exponential
func QExp(q Quantity) Quantity {
	return &expOp{q}
}

type expOp struct{ orig Quantity }

func (op *expOp) NComp() int { return op.orig.NComp() }
func (op *expOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QExp(dst, src)
}

// QLog: pointwise natural logarithm
func QLog(q Quantity) Quantity {
	return &logOp{q}
}

type logOp struct{ orig Quantity }

func (op *logOp) NComp() int { return op.orig.NComp() }
func (op *logOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QLog(dst, src)
}

// QAbs: pointwise absolute value
func QAbs(q Quantity) Quantity {
	return &absOp{q}
}

type absOp struct{ orig Quantity }

func (op *absOp) NComp() int { return op.orig.NComp() }
func (op *absOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAbs(dst, src)
}

// QAcos: pointwise arccos
func QAcos(q Quantity) Quantity {
	return &acosOp{q}
}

type acosOp struct{ orig Quantity }

func (op *acosOp) NComp() int { return op.orig.NComp() }
func (op *acosOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAcos(dst, src)
}

// QAcosh: pointwise inverse hyperbolic cosine
func QAcosh(q Quantity) Quantity {
	return &acoshOp{q}
}

type acoshOp struct{ orig Quantity }

func (op *acoshOp) NComp() int { return op.orig.NComp() }
func (op *acoshOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAcosh(dst, src)
}

// QAsin: pointwise arcsine
func QAsin(q Quantity) Quantity {
	return &asinOp{q}
}

type asinOp struct{ orig Quantity }

func (op *asinOp) NComp() int { return op.orig.NComp() }
func (op *asinOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAsin(dst, src)
}

// QAsinh: pointwise inverse hyperbolic sine
func QAsinh(q Quantity) Quantity {
	return &asinhOp{q}
}

type asinhOp struct{ orig Quantity }

func (op *asinhOp) NComp() int { return op.orig.NComp() }
func (op *asinhOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAsinh(dst, src)
}

// QAtan: pointwise arctangent
func QAtan(q Quantity) Quantity {
	return &atanOp{q}
}

type atanOp struct{ orig Quantity }

func (op *atanOp) NComp() int { return op.orig.NComp() }
func (op *atanOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAtan(dst, src)
}

// QAtanh: pointwise inverse hyperbolic tangent
func QAtanh(q Quantity) Quantity {
	return &atanhOp{q}
}

type atanhOp struct{ orig Quantity }

func (op *atanhOp) NComp() int { return op.orig.NComp() }
func (op *atanhOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QAtanh(dst, src)
}

// QCosh: pointwise hyperbolic cosine
func QCosh(q Quantity) Quantity {
	return &coshOp{q}
}

type coshOp struct{ orig Quantity }

func (op *coshOp) NComp() int { return op.orig.NComp() }
func (op *coshOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QCosh(dst, src)
}

// QSinh: pointwise hyperbolic sine
func QSinh(q Quantity) Quantity {
	return &sinhOp{q}
}

type sinhOp struct{ orig Quantity }

func (op *sinhOp) NComp() int { return op.orig.NComp() }
func (op *sinhOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QSinh(dst, src)
}

// QTan: pointwise tangent
func QTan(q Quantity) Quantity {
	return &tanOp{q}
}

type tanOp struct{ orig Quantity }

func (op *tanOp) NComp() int { return op.orig.NComp() }
func (op *tanOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QTan(dst, src)
}

// QTanh: pointwise hyperbolic tangent
func QTanh(q Quantity) Quantity {
	return &tanhOp{q}
}

type tanhOp struct{ orig Quantity }

func (op *tanhOp) NComp() int { return op.orig.NComp() }
func (op *tanhOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QTanh(dst, src)
}

// QErf: pointwise error function
func QErf(q Quantity) Quantity {
	return &erfOp{q}
}

type erfOp struct{ orig Quantity }

func (op *erfOp) NComp() int { return op.orig.NComp() }
func (op *erfOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QErf(dst, src)
}

// QErfc: pointwise complementary error function
func QErfc(q Quantity) Quantity {
	return &erfcOp{q}
}

type erfcOp struct{ orig Quantity }

func (op *erfcOp) NComp() int { return op.orig.NComp() }
func (op *erfcOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QErfc(dst, src)
}

// QGamma: pointwise gamma function
func QGamma(q Quantity) Quantity {
	return &gammaOp{q}
}

type gammaOp struct{ orig Quantity }

func (op *gammaOp) NComp() int { return op.orig.NComp() }
func (op *gammaOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QGamma(dst, src)
}

// QHeaviside: pointwise Heaviside step function
func QHeaviside(q Quantity) Quantity {
	return &heavisideOp{q}
}

type heavisideOp struct{ orig Quantity }

func (op *heavisideOp) NComp() int { return op.orig.NComp() }
func (op *heavisideOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QHeaviside(dst, src)
}

// QSinc: pointwise normalized sinc function
func QSinc(q Quantity) Quantity {
	return &sincOp{q}
}

type sincOp struct{ orig Quantity }

func (op *sincOp) NComp() int { return op.orig.NComp() }
func (op *sincOp) EvalTo(dst *data.Slice) {
	src := ValueOf(op.orig)
	defer cuda.Recycle(src)
	cuda.QSinc(dst, src)
}

// QMod: pointwise modulo (two-argument)
func QMod(a, b Quantity) Quantity {
	return &modOp{a, b}
}

type modOp struct{ a, b Quantity }

func (op *modOp) NComp() int {
	if op.a.NComp() >= op.b.NComp() {
		return op.a.NComp()
	}
	return op.b.NComp()
}

func (op *modOp) EvalTo(dst *data.Slice) {
	A := ValueOf(op.a)
	defer cuda.Recycle(A)
	B := ValueOf(op.b)
	defer cuda.Recycle(B)
	cuda.QMod(dst, A, B)
}

// QPow: pointwise power (two-argument)
func QPow(a, b Quantity) Quantity {
	return &powOp{a, b}
}

type powOp struct{ a, b Quantity }

func (op *powOp) NComp() int {
	if op.a.NComp() >= op.b.NComp() {
		return op.a.NComp()
	}
	return op.b.NComp()
}

func (op *powOp) EvalTo(dst *data.Slice) {
	A := ValueOf(op.a)
	defer cuda.Recycle(A)
	B := ValueOf(op.b)
	defer cuda.Recycle(B)
	cuda.QPow(dst, A, B)
}
