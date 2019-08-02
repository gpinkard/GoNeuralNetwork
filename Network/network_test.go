package GoNeuralNetwork

import (
	"reflect"
	"testing"
)

func TestCreateNetwork(t *testing.T) {
	net := CreateNetwork()
	if reflect.TypeOf(net).String() != "*GoNeuralNetwork.network" {
		t.Errorf("Expected type of *GoNeuralNetwork.network but got %s", reflect.TypeOf(net).String())
	}
}

func TestAddLayer(t *testing.T) {
	net := CreateNetwork()
	net.AddLayer("basiclayer", "sigmoid", 4)
	net.AddLayer("basiclayer", "sigmoid", 4)

	if len(net.layers) != 2 {
		t.Errorf("Expected network to have two layers, but it has %d", len(net.layers))
	}

	// test to see if they connected correctly
	TestConnectLayers(t)
}

func TestCalculateNetworkOutput(t *testing.T) {
	net := CreateNetwork()
	inputData := []float64{0.5, 0.5, 0.5}

	_, err := net.CalculateNetworkOutput(inputData)
	if err == nil {
		t.Errorf("Expected error to be returned because network has not layers, but got nil value")
	}

	net.AddLayer("basiclayer", "sigmoid", 4)
	//net.AddLayer("basiclayer", "sigmoid", 4)
	_, err = net.CalculateNetworkOutput(inputData)
	if err == nil {
		t.Errorf("Expected error to be returned because input data and input layer are not the same length, but got nil value")
	}

	inputData = append(inputData, 0.5)
	res, _ := net.CalculateNetworkOutput(inputData)
	lastLayerLength := len(net.layers[len(net.layers)-1].getNeurons())
	if len(res) != lastLayerLength {
		t.Errorf("Returned slice not the same length as the last layer in network, expected %d, got %d", lastLayerLength, len(res))
	}
}
