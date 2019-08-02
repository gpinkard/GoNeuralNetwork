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
}

func TestConnectLayers(t *testing.T) {
	l1, _ := createLayer("basiclayer", "sigmoid", 3)
	l2, _ := createLayer("basiclayer", "sigmoid", 3)
	connectLayers(l1, l2)

	for _, n1 := range l1.getNeurons() {
		for _, n2 := range l2.getNeurons() {
			foundN1, foundN2 := false, false
			for _, w := range n1.weightsIn {
				if w.from == n2 {
					foundN2 = true
					break
				}
			}
			for _, w := range n2.weightsOut {
				if w.to == n1 {
					foundN1 = true
					break
				}
			}
			if !(foundN1 && foundN2) {
				t.Errorf("not all neurons between two test layers connected correctly...")
			}
		}
	}
}
