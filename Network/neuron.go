package GoNeuralNetwork

import "sync"

/*
A structure that represents a neuron
*/
type neuron struct {
	output     float64             // the output of this neuron
	weightsIn  map[*neuron]float64 // weights coming into this neuron
	weightsOut map[*neuron]float64 // weights from this neuron to other neurons
	deltas     map[*neuron]float64 // delta values (back propagation)
	deltaMutex sync.RWMutex        // mutex lock for deltas map
	//weightsMutex sync.RWMutex        // weightsIn mutex
	af actfunc // activation function
}

/*
Returns a pointer to a neuron with the specified activation function and output
*/
func createNeuron(af actfunc, output float64) *neuron {
	return &neuron{
		output:     output,
		weightsIn:  make(map[*neuron]float64),
		weightsOut: make(map[*neuron]float64),
		deltas:     make(map[*neuron]float64),
		af:         af,
	}
}

/*
Calculates the output of this neuron
*/
func (n *neuron) calcOutput() {
	ws := 0.0
	for nFrom, w := range n.weightsIn {
		ws += (w * nFrom.output)
	}
	n.output = n.af.calculate(ws)
}

/*
Gets the delta values for connections to this neuron
*/
func (n *neuron) getDeltas(trainingDatum float64, isHiddenLayer bool) {
	if isHiddenLayer {
		for nTo, w := range n.weightsOut {
			n.deltaMutex.RLock()
			n.deltas[nTo] = n.af.calculateDelta(w, n.output, trainingDatum, n.deltas[nTo], isHiddenLayer)
			n.deltaMutex.RUnlock()
		}
	} else {
		for nFrom, w := range n.weightsIn {
			n.deltaMutex.RLock()
			n.deltas[nFrom] = n.af.calculateDelta(w, n.output, trainingDatum, n.deltas[nFrom], isHiddenLayer)
			n.deltaMutex.RUnlock()
		}
	}
}

/*
Update the weights of the connections into this neuron from corresponding delta values
*/
func (n *neuron) updateWeights() {
	for nTo, delta := range n.deltas {
		updatedWeight := n.weightsOut[nTo] + (1 * delta * n.output)
		n.weightsOut[nTo] = updatedWeight
		nTo.weightsIn[n] = updatedWeight
	}
}
