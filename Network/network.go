package GoNeuralNetwork

import (
	"errors"
	"fmt"
	"log"
	"sync"
)

/*
Structure representing the neural network, a slice of type layer
*/
type network struct {
	layers []layer
}

/*
Returns a pointer to a new network structure
*/
func CreateNetwork() *network {
	return &network{
		layers: make([]layer, 0),
	}
}

/*
Adds a layer to the neural network of the given layer and activation function type, and the
specified size.
*/
func (net *network) AddLayer(layerType, actfunc string, size int) {
	l, err := createLayer(layerType, actfunc, size)
	if err != nil {
		log.Panic(err)
	}

	net.layers = append(net.layers, l)

	// if this is not the first layer connect to previous layer
	if len(net.layers) != 1 {
		layerFrom := net.layers[len(net.layers)-2]
		layerTo := net.layers[len(net.layers)-1]
		connectLayers(layerFrom, layerTo)
	}
}

/*
Calculates the output of the neural network given a slice of input data. Returns a non-nil error
if there are no layers in the network, or if the input slice and the first layer are not of the
same size. Returns a slice of outputs corresponding to the last layers output values (i.e. if
the last layer has ten elements, the returned slice will have ten elements, each of which corresponds
to the index of the neuron in the layer that produced it).
*/
func (net *network) CalcNetworkOutput(input []float64) ([]float64, error) {
	if len(net.layers) == 0 {
		return nil, errors.New("There are no layers in this network, cannot calculate output...")
	}

	if len(input) != len(net.layers[0].getNeurons()) {
		return nil, errors.New("Input data slice not same length as input layer of network...")
	}

	outputs := make([]float64, len(input)) // outputs of last layers neurons will be put in this slice
	for i, l := range net.layers {
		for j, n := range l.getNeurons() {
			// if this is the first layer we don't need to calculate our output
			if i == 0 {
				n.output = input[j]
				continue
			}
			n.calcOutput()
			// if this is the last layer, set entry in output to corresponding neuron output
			if i == len(net.layers)-1 {
				outputs[j] = n.output
			}
		}
	}

	return outputs, nil
}

/*
Train the neural network. Takes in a 2D slice of training data, and a learning rate constant.
Trains for the specified number of epochs on each training element. Returns a non-nill error
if the trainging data slice is not the same length as the same length as input layer, or if
the epoch number is less than or equal to zero.
*/
func (net *network) Train(trainingDataSet [][]float64, learningRate float64, epochs int) error {
	if epochs <= 0 {
		return errors.New("epochs must be greater or equal to zero...")
	}

	for _, td := range trainingDataSet {
		_, err := net.CalcNetworkOutput(td)
		if err != nil {
			log.Panic(err)
		}
		for i := 0; i < epochs; i++ {
			net.doBackPropagation(td, learningRate)
		}
	}

	return nil
}

/*
Performs the back propagation training algorithm. Takes slice of training data and a learning
rate constant.
*/
func (net *network) doBackPropagation(trainingData []float64, learningRate float64) {
	var wg sync.WaitGroup

	for i := 0; i < len(trainingData); i++ {
		for j := len(net.layers) - 1; j >= 0; j-- {
			// true if this is not the output layer
			isHiddenLayer := j == len(net.layers)-1

			for ind, n := range net.layers[j].getNeurons() {
				wg.Add(1)
				go func(n *neuron, trainingDatum float64, isHiddenLayer bool) {
					defer wg.Done()
					n.getDeltas(trainingDatum, isHiddenLayer)
				}(n, trainingData[ind], isHiddenLayer)
			}
			wg.Wait() // wait for all these go routines to finish
		}

		// update weights now that deltas have been calculated
		for _, l := range net.layers {
			for _, n := range l.getNeurons() {
				n.updateWeights()
				/*
					TODO fix race condition when writing to two maps concurently in updateWeights
						wg.Add(1)
						go func(n *neuron) {
							defer wg.Done()
							n.updateWeights()
						}(n)
				*/
			}
			//wg.Wait()
		}
	}
}

// prints network, for debugging...
func (net *network) PrintNetwork() {
	fmt.Println("----- PRINTING NETWORK -----")
	for i, l := range net.layers {
		fmt.Println("LAYER", i)
		for _, n := range l.getNeurons() {
			fmt.Printf("%p, output: %f, IN:", n, n.output)
			fmt.Println(n.weightsIn, "OUT: ", n.weightsOut)
		}
		fmt.Println("")
	}
}
