package GoNeuralNetwork

import (
	"errors"
	"math/rand"
	"time"
)

/*
An interface that represents a layer in a neural network
*/
type layer interface {
	// return the slice of neurons in the layer
	getNeurons() []*neuron
}

// for testing purposes only, will add more later
type basicLayer struct {
	neurons []*neuron
}

/*
Creates and returns a layer given a specified layer type, activation function and
size. Returns a non-nil error if the activation function or layer type is not valid,
or if the size is less than 1.
*/
func createLayer(layerType, actfuncType string, size int) (layer, error) {
	af, err := createActFunc(actfuncType)
	if err != nil {
		return nil, err
	}

	if size <= 0 {
		return nil, errors.New("specified size must be greater or equal to 1")
	}

	xn := make([]*neuron, size)
	for i := 0; i < size; i++ {
		xn[i] = createNeuron(af)
	}

	switch layerType {
	case "basiclayer":
		return &basicLayer{
			neurons: xn,
		}, nil
	default:
		return nil, errors.New(layerType + " is not a valid kind of layer")
	}
}

/*
Connects the neurons between the two and from layers (creates weights between neurons)
*/
func connectLayers(to, from layer) {
	rand.Seed(time.Now().UnixNano())
	for _, nTo := range to.getNeurons() {
		for _, nFrom := range from.getNeurons() {
			// for now, init values to values between 0.0 and 0.99
			w := createWeight(nTo, nFrom, rand.Float64())
			nTo.weightsIn = append(nTo.weightsIn, w)
			nFrom.weightsOut = append(nFrom.weightsOut, w)
		}
	}
}

func (bl *basicLayer) getNeurons() []*neuron {
	return bl.neurons
}
