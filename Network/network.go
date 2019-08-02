package GoNeuralNetwork

import (
	"errors"
	"fmt"
	"log"
	"sync"
)

type network struct {
	layers []layer
}

func CreateNetwork() *network {
	return &network{
		layers: make([]layer, 0),
	}
}

func (net *network) AddLayer(layerType, actfuncType string, size int) {
	l, err := createLayer(layerType, actfuncType, size)
	if err != nil {
		log.Panic(err)
	}

	net.layers = append(net.layers, l)

	if len(net.layers) != 1 {
		layerFrom := net.layers[len(net.layers)-2]
		layerTo := net.layers[len(net.layers)-1]
		connectLayers(layerTo, layerFrom)
	}
}

func (net *network) CalculateNetworkOutput(input []float64) ([]float64, error) {
	if len(net.layers) == 0 {
		return nil, errors.New("there are no layers in this network, cannot calculate the networks output...")
	}

	if len(input) != len(net.layers[0].getNeurons()) {
		return nil, errors.New("input data slice not same length as input layer of network")
	}

	outputs := make([]float64, len(net.layers[len(net.layers)-1].getNeurons()))

	for i, l := range net.layers {
		for j, n := range l.getNeurons() {
			if i == 0 {
				n.output = input[j]
				continue
			}
			n.calcOutput()
			if i == len(net.layers)-1 {
				outputs[j] = n.output
			}
		}
	}
	return outputs, nil
}

func (net *network) Train(trainingDataSet [][]float64, learningRate float64, epochs int) error {
	if epochs <= 0 {
		return errors.New("specified epochs must be greater or equal to 0...")
	}

	for _, td := range trainingDataSet {
		_, err := net.CalculateNetworkOutput(td)
		if err != nil {
			log.Panic(err)
		}
		for i := 0; i < epochs; i++ {
			net.doBackPropagation(td, learningRate)
		}
	}

	return nil
}

func (net *network) doBackPropagation(trainingData []float64, learningRate float64) {
	var wg sync.WaitGroup

	for i := 0; i < len(trainingData); i++ {
		for j := len(net.layers) - 1; j >= 0; j-- {
			isHiddenLayer := j == len(net.layers)-1

			for ind, n := range net.layers[j].getNeurons() {
				wg.Add(1)
				go func(n *neuron, trainingDatum float64, isHiddenLayer bool) {
					defer wg.Done()
					n.getDeltas(trainingDatum, isHiddenLayer)
				}(n, trainingData[ind], isHiddenLayer)
			}
			wg.Wait()
		}

		for _, l := range net.layers {
			for _, n := range l.getNeurons() {
				wg.Add(1)
				go func(n *neuron) {
					defer wg.Done()
					n.updateWeights()
				}(n)
			}
			wg.Wait()
		}
	}
}

func (net *network) PrintNetwork() {
	fmt.Println("----- PRINTING NETWORK -----")
	for i, l := range net.layers {
		fmt.Printf("LAYER %d\n\n", i)
		for _, n := range l.getNeurons() {
			fmt.Printf("neuron: %p, output: %f\n\n", n, n.output)
			fmt.Printf("\tIN:\n")
			for _, w := range n.weightsIn {
				fmt.Printf("\t%p: %f\n", w.from, w.w)
			}
			fmt.Printf("\tOUT:\n")
			for _, w := range n.weightsOut {
				fmt.Printf("\t%p: %f\n", w.to, w.w)
			}
			fmt.Println("")
		}
	}
}
