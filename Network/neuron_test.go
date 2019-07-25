package GoNeuralNetwork

import "testing"

func TestCreateNeuron(t *testing.T) {
	af, _ := createActFunc("sigmoid")
	n := createNeuron(af, 0.5)

	if n.af.getActFunc() != "sigmoid" {
		t.Errorf("Expected activation function of type sigmoid but got " + n.af.getActFunc())
	}

	if n.output != 0.5 {
		t.Errorf("Expected value of 0.5, but got %f", n.output)
	}

	if len(n.weightsIn) != 0 {
		t.Errorf("Expected empty weightsIn map, but size is %d", len(n.weightsIn))
	}

	if len(n.weightsOut) != 0 {
		t.Errorf("Expected empty weightsOut map, but size is %d", len(n.weightsOut))
	}

	if len(n.deltas) != 0 {
		t.Errorf("Expected empty deltas map, but size is %d", len(n.deltas))
	}
}

func TestCalcOutput(t *testing.T) {
	af, _ := createActFunc("relu")

	n1 := createNeuron(af, 0.5)
	n2 := createNeuron(af, 0.5)
	n3 := createNeuron(af, 0.5)
	n1.weightsOut[n3] = 0.5
	n2.weightsOut[n3] = 0.5
	n3.weightsIn[n2] = 0.5
	n3.weightsIn[n1] = 0.5
	n3.calcOutput()

	if n3.output != 0.5 {
		t.Errorf("Expected output of 0.5, but got %f", n3.output)
	}
}

func TestUpdateWeights(t *testing.T) {
	af, _ := createActFunc("sigmoid")

	n1 := createNeuron(af, 0.5)
	n2 := createNeuron(af, 0.5)
	n1.weightsOut[n2] = 0.5
	n2.weightsIn[n1] = 0.5
	n1.deltas[n2] = 0.5

	n1.updateWeights()
	if n1.weightsOut[n2] != 0.75 {
		t.Errorf("expected weight from n1 to n2 to be 0.75 but got %f", n1.weightsOut[n2])
	}

	if n2.weightsIn[n1] != 0.75 {
		t.Errorf("expected weight from n1 to n2 to be 0.75 but got %f", n1.weightsOut[n2])
	}
}
