package GoNeuralNetwork

import (
	"fmt"
	"strconv"
	"testing"
)

func TestCreateWeight(t *testing.T) {
	af, _ := createActFunc("sigmoid")
	n1 := createNeuron(af)
	n2 := createNeuron(af)
	w := createWeight(n1, n2, 0.5)

	n1Str := fmt.Sprintf("%v", n1)
	n2Str := fmt.Sprintf("%v", n2)
	wtoStr := fmt.Sprintf("%v", w.to)
	wfromStr := fmt.Sprintf("%v", w.from)

	if w.w != 0.5 {
		t.Errorf("expected weight to be 0.5, but got " + strconv.FormatFloat(w.w, 'f', 3, 64))
	}

	if n1 != w.to {
		t.Errorf("expected to neuron to be " + n1Str + " but got " + wtoStr)
	}

	if n2 != w.from {
		t.Errorf("expected to neuron to be " + n2Str + " but got " + wfromStr)
	}
}
