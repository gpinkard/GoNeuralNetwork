package GoNeuralNetwork

import (
	"reflect"
	"testing"
)

func TestCreateLayer(t *testing.T) {
	_, err := createLayer("notalayer", "sigmoid", 10)
	if err == nil {
		t.Errorf("Expected error, but nil value returned")
	}
	_, err = createLayer("basiclayer", "notanactfunc", 10)
	if err == nil {
		t.Errorf("Expected error to be returned from createActFunc call, but nil value returned")
	}
}

func TestGetNeurons(t *testing.T) {
	bl, _ := createLayer("basiclayer", "sigmoid", 10)
	if reflect.TypeOf(bl).String() != "*GoNeuralNetwork.basicLayer" {
		t.Errorf("Expected layer to be of type basicLayer but got %s", reflect.TypeOf(bl).String())
	}
	if len(bl.getNeurons()) != 10 {
		t.Errorf("Expected length of neuron slice to be 10, but got %d", len(bl.getNeurons()))
	}

	il, _ := createLayer("inputlayer", "sigmoid", 10)
	if reflect.TypeOf(il).String() != "*GoNeuralNetwork.inputLayer" {
		t.Errorf("Expected layer to be of type inputLayer but got %s", reflect.TypeOf(il).String())
	}
	if len(il.getNeurons()) != 10 {
		t.Errorf("Expected length of neuron slice to be 10, but got %d", len(bl.getNeurons()))
	}
}

func TestConnectLayers(t *testing.T) {
	l1, _ := createLayer("basiclayer", "sigmoid", 3)
	l2, _ := createLayer("basiclayer", "sigmoid", 3)
	connectLayers(l1, l2)

	for _, n1 := range l1.getNeurons() {
		for _, n2 := range l2.getNeurons() {
			if _, ok1 := n1.weightsOut[n2]; ok1 {
				if _, ok2 := n2.weightsIn[n1]; ok2 {
					continue
				} else {
					t.Errorf("Found two neurons in connected layers that are not connected")
				}
			}
			t.Errorf("Found two neurons in connected layers that are not connected")
		}
	}
}
