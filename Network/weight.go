package GoNeuralNetwork

/*
A struct representing a weight between two neurons.
*/
type weight struct {
	// neuron thats being fed into by this weight
	to *neuron
	// neuron this weight is leading out of
	from *neuron
	// the weight itself
	w float64
}

/*
Creates and returns a weight given two pointers to neurons and a weight value.
*/
func createWeight(to, from *neuron, w float64) *weight {
	return &weight{
		to:   to,
		from: from,
		w:    w,
	}
}
