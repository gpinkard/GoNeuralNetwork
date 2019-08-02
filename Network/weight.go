package GoNeuralNetwork

type weight struct {
	to   *neuron
	from *neuron
	w    float64 // the weight itself
}

func createWeight(to, from *neuron, w float64) *weight {
	return &weight{
		to:   to,
		from: from,
		w:    w,
	}
}
