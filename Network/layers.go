package GoNeuralNetwork

import (
	"errors"
	"math/rand"
	"strings"
	"time"
)

/*
An interface for different layer types. A layer is a __pointer__ to a struct
*/
type layer interface {
	// return the slice of neurons this layer contains
	getNeurons() []*neuron
}

// mostly for testing purposes
type basicLayer struct {
	neurons []*neuron
}

// TODO not used currently
type inputLayer struct {
	neurons []*neuron
}

/*
Creates a layer given two strings specifying layer type and activation function, as
well as a size. Returns a layer, and a non-nil error if the activation function or
layer type strings are not recognized
*/
func createLayer(layerType, actfunc string, size int) (layer, error) {
	layerType = strings.ToLower(layerType)
	af, err := createActFunc(actfunc)
	if err != nil {
		return nil, err
	}

	rand.Seed(time.Now().UnixNano())

	xn := make([]*neuron, size)
	for i := 0; i < size; i++ {
		xn[i] = createNeuron(af, rand.Float64())
	}

	switch layerType {
	case "basiclayer":
		return &basicLayer{
			neurons: xn,
		}, nil
	case "inputlayer":
		return &inputLayer{
			neurons: xn,
		}, nil
	}

	return nil, errors.New(layerType + "is not a valid layer type")
}

/*
Connects the from layer to the to layer (currently, all neurons in these layers will
have random weights assigned between 0.0 and .99)
*/
func connectLayers(from, to layer) {
	rand.Seed(time.Now().UnixNano())
	weight := 0.0
	for _, nFrom := range from.getNeurons() {
		for _, nTo := range to.getNeurons() {
			weight = rand.Float64()
			nFrom.weightsOut[nTo] = weight
			nTo.weightsIn[nFrom] = weight
		}
	}
}

func (bl *basicLayer) getNeurons() []*neuron {
	return bl.neurons
}

func (il *inputLayer) getNeurons() []*neuron {
	return il.neurons
}
