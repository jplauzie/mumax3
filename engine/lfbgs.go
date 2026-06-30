package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

const (
	lbfgsHistory = 5
)

type LBFGSMinimizer struct {

	// current gradient
	g *data.Slice

	// previous gradient
	gOld *data.Slice

	// search direction / q vector
	q *data.Slice

	// previous magnetization
	xOld *data.Slice

	// temporary vectors
	s *data.Slice
	y *data.Slice

	// limited-memory history
	sHist [lbfgsHistory]*data.Slice
	yHist [lbfgsHistory]*data.Slice

	alpha [lbfgsHistory]float32

	H0 float32
	f  float64

	initialized bool

	iter     int
	globIter int
	rho      [lbfgsHistory]float32

	lastDm fifoRing
}

func NewLBFGSMinimizer() *LBFGSMinimizer {

	size := M.Buffer().Size()

	m := &LBFGSMinimizer{
		g:      cuda.Buffer(3, size),
		gOld:   cuda.Buffer(3, size),
		q:      cuda.Buffer(3, size),
		xOld:   cuda.Buffer(3, size),
		s:      cuda.Buffer(3, size),
		y:      cuda.Buffer(3, size),
		H0:     1,
		lastDm: FifoRing(DmSamples),
	}

	for i := 0; i < lbfgsHistory; i++ {
		m.sHist[i] = cuda.Buffer(3, size)
		m.yHist[i] = cuda.Buffer(3, size)
	}

	return m
}

func (m *LBFGSMinimizer) Free() {

	m.g.Free()
	m.gOld.Free()
	m.q.Free()
	m.xOld.Free()

	m.s.Free()
	m.y.Free()

	for i := range m.sHist {
		m.sHist[i].Free()
		m.yHist[i].Free()
	}
}
