package GoNeuralNetwork

import "sync"

/*
Struct representing a neuron in the neural network.
*/
type neuron struct {
	output          float64      // output of this neuron
	delta           float64      // delta value of this neuron
	weightsIn       []*weight    // weights leading into this neuron
	weightsOut      []*weight    // weights leading out of this neuron
	weightsInMutex  sync.RWMutex // mutex lock for weightsIn
	weightsOutMutex sync.RWMutex // mutex lock for weightsOut
	af              actfunc      // neurons activation function
}

/*
Creates and returns a pointer to a neuron with the specified activation function.
*/
func createNeuron(af actfunc) *neuron {
	return &neuron{
		output:     0.0,
		delta:      0.0,
		weightsIn:  make([]*weight, 0),
		weightsOut: make([]*weight, 0),
		af:         af,
	}
}

/*
Calculates the output of the given neuron.
*/
func (n *neuron) calcOutput() {
	ws := 0.0
	for _, w := range n.weightsIn {
		ws += (w.w * w.from.output)
	}
	n.output = n.af.calculate(ws)
}

/*
Gets the delta value for the specified neuron. Takes a training datum, and a boolean (true
if the layer is a hidden layer).
*/
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

/*
Update the weights of the weights leaving this neuron based on this neurons delta value.
*/
func (n *neuron) updateWeights() {
	for _, w := range n.weightsOut {
		w.w = w.w + (w.to.delta * n.output)
	}
}
