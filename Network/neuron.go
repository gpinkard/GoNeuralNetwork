package GoNeuralNetwork

import "sync"

type neuron struct {
	output          float64
	delta           float64
	weightsIn       []*weight
	weightsOut      []*weight
	weightsInMutex  sync.RWMutex
	weightsOutMutex sync.RWMutex
	af              actfunc
}

func createNeuron(af actfunc) *neuron {
	return &neuron{
		output:     0.0,
		delta:      0.0,
		weightsIn:  make([]*weight, 0),
		weightsOut: make([]*weight, 0),
		af:         af,
	}
}

func (n *neuron) calcOutput() {
	ws := 0.0
	for _, w := range n.weightsIn {
		ws += (w.w * w.from.output)
	}
	n.output = n.af.calculate(ws)
}

func (n *neuron) getDeltas(trainingDatum float64, isHiddenLayer bool) {
	if isHiddenLayer {
		for _, w := range n.weightsOut {
			n.weightsOutMutex.RLock()
			n.delta = n.af.calculateDelta(w.w, n.output, trainingDatum, n.delta, isHiddenLayer)
			n.weightsOutMutex.RUnlock()
		}
	} else {
		for _, w := range n.weightsIn {
			n.weightsInMutex.RLock()
			n.delta = n.af.calculateDelta(w.w, n.output, trainingDatum, n.delta, isHiddenLayer)
			n.weightsInMutex.RUnlock()
		}
	}
}

// when I update my weights, the delta comes from the guy ahead of me
func (n *neuron) updateWeights() {
	for _, w := range n.weightsOut {
		w.w = w.w + (w.to.delta * n.output)
	}
}
