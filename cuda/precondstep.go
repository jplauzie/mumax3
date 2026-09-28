package cuda

import "github.com/mumax/3/data"

// PrecondStep updates the running average of k^2 and computes the
// preconditioned direction p = k / (sqrt(avg) + eps).
// decay = 0 initialises avg from the current k.
func PrecondStep(p, avg, k *data.Slice, decay, eps float32) {
	N := k.Len()
	cfg := make1DConf(N)
	k_precond_step_async(
		p.DevPtr(X), p.DevPtr(Y), p.DevPtr(Z),
		avg.DevPtr(X),
		k.DevPtr(X), k.DevPtr(Y), k.DevPtr(Z),
		decay, eps, N, cfg)
}
