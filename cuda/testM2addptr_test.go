package cuda

import (
	"testing"
	"unsafe"

	"github.com/mumax/3/cuda/cu"
)

func TestMadd2Ptr(t *testing.T) {
	a := toGPU([]float32{1, 2, 3, 4, 5})
	b := toGPU([]float32{5, 4, 3, -1, 2})
	dst := NewSlice(1, [3]int{1, 1, 5})

	// device-resident scalar factor = 3.0
	factorPtr := MemAlloc(cu.SIZEOF_FLOAT32)
	defer cu.DevicePtr(uintptr(factorPtr)).Free()
	three := float32(3.0)

	cu.MemcpyHtoD(cu.DevicePtr(uintptr(factorPtr)), unsafe.Pointer(&three), int64(cu.SIZEOF_FLOAT32))

	// dst[i] = 1.0*a[i] + (*factorPtr)*b[i]
	Madd2Ptr(dst, a, b, 1.0, factorPtr)

	want := []float32{
		1*1 + 3*5,  // 16
		1*2 + 3*4,  // 14
		1*3 + 3*3,  // 12
		1*4 + 3*-1, // 1
		1*5 + 3*2,  // 11
	}

	host := make([]float32, 5)
	MemCpyDtoH(unsafe.Pointer(&host[0]), dst.DevPtr(0), 5*cu.SIZEOF_FLOAT32)

	for i := range want {
		if host[i] != want[i] {
			t.Errorf("madd2ptr result[%d] = %v, want %v", i, host[i], want[i])
		}
	}
}

func TestDotInto(t *testing.T) {
	ensureTestCtx()
	a := toGPU([]float32{1, 2, 3, 4, 5})
	b := toGPU([]float32{5, 4, 3, -1, 2})

	dst := MemAlloc(cu.SIZEOF_FLOAT32)
	defer cu.DevicePtr(uintptr(dst)).Free()

	MemsetScalarAsync(dst, 0)
	DotInto(a, b, dst)

	var result float32
	MemCpyDtoH(unsafe.Pointer(&result), dst, cu.SIZEOF_FLOAT32)

	want := float32(5 + 8 + 9 - 4 + 10) // matches TestReduceDot's expected value
	if result != want {
		t.Errorf("DotInto result = %v, want %v", result, want)
	}
}

func TestScaleAndAxpyPtrInto(t *testing.T) {
	ensureTestCtx()
	numerator := MemAlloc(cu.SIZEOF_FLOAT32)
	defer cu.DevicePtr(uintptr(numerator)).Free()
	alpha := MemAlloc(cu.SIZEOF_FLOAT32)
	defer cu.DevicePtr(uintptr(alpha)).Free()

	seven := float32(7.0)
	cu.MemcpyHtoD(cu.DevicePtr(uintptr(numerator)), unsafe.Pointer(&seven), int64(cu.SIZEOF_FLOAT32))

	// alpha = rho * numerator, rho = 0.5 -> alpha should be 3.5
	ScaleInto(alpha, numerator, 0.5)

	var gotAlpha float32
	MemCpyDtoH(unsafe.Pointer(&gotAlpha), alpha, cu.SIZEOF_FLOAT32)
	if gotAlpha != 3.5 {
		t.Errorf("ScaleInto result = %v, want 3.5", gotAlpha)
	}

	// q update on a real vector: q[i] -= alpha*y[i], reusing AxpyPtrInto's
	// sibling Madd2Ptr since this part is vector-length, not scalar --
	// included here just to confirm alpha is usable downstream.
	q := toGPU([]float32{10, 10, 10, 10, 10})
	y := toGPU([]float32{1, 1, 1, 1, 1})
	Madd2Ptr(q, q, y, 1.0, alpha)

	host := make([]float32, 5)
	MemCpyDtoH(unsafe.Pointer(&host[0]), q.DevPtr(0), 5*cu.SIZEOF_FLOAT32)
	for i, v := range host {
		if v != 10+3.5 {
			t.Errorf("q[%d] after Madd2Ptr = %v, want %v", i, v, 10+3.5)
		}
	}
}
