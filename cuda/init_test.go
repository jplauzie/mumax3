package cuda

import (
	"runtime"

	"github.com/mumax/3/cuda/cu"
)

var testCtx cu.Context

// needed for all other tests.
func init() {
	cu.Init(0)
	testCtx = cu.CtxCreate(cu.CTX_SCHED_AUTO, 0)
	cu.CtxSetCurrent(testCtx)
}

// ensureTestCtx locks the calling goroutine to its OS thread and makes
// the shared test CUDA context current on that thread. CUDA contexts are
// thread-local, but Go test functions each run in their own goroutine
// which may be scheduled onto any OS thread -- without this, a test can
// land on a thread where no context is current, causing
// CUDA_ERROR_INVALID_CONTEXT.
func ensureTestCtx() {
	runtime.LockOSThread()
	cu.CtxSetCurrent(testCtx)
}
