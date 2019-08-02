package GoNeuralNetwork

import "testing"

func TestCreateNeuron(t *testing.T) {
	af, _ := createActFunc("sigmoid")
	n := createNeuron(af)

	if n.af.getActFunc() != "sigmoid" {
		t.Errorf("Expected activation function of type sigmoid but got " + n.af.getActFunc())
	}

	if len(n.weightsIn) != 0 {
		t.Errorf("Expected empty weightsIn map, but size is %d", len(n.weightsIn))
	}

	if len(n.weightsOut) != 0 {
		t.Errorf("Expected empty weightsOut map, but size is %d", len(n.weightsOut))
	}
}

func TestCalcOutput(t *testing.T) {
	af, _ := createActFunc("relu")

	n1 := createNeuron(af)
	n2 := createNeuron(af)
	n3 := createNeuron(af)
	n1.output, n2.output, n3.output = 0.5, 0.5, 0.5
	w1 := createWeight(n1, n2, 0.5)
	w2 := createWeight(n2, n3, 0.5)
	n1.weightsOut = append(n1.weightsOut, w1)
	n2.weightsIn = append(n1.weightsIn, w1)
	n2.weightsOut = append(n2.weightsOut, w2)
	n3.weightsIn = append(n3.weightsIn, w2)
	n3.calcOutput()

	if n3.output != 0.25 {
		t.Errorf("Expected output of 0.25, but got %f", n3.output)
	}
}

func TestUpdateWeights(t *testing.T) {
	af, _ := createActFunc("sigmoid")

	n1 := createNeuron(af)
	n2 := createNeuron(af)
	w := createWeight(n1, n2, 0.5)
	n2.delta = 0.5
	n1.weightsOut = append(n1.weightsOut, w)
	n2.weightsIn = append(n2.weightsIn, w)

	n1.updateWeights()

	if n1.weightsOut[0].w != 0.5 {
		t.Errorf("expected weight from n1 to n2 to be 0.5 but got %f", n1.weightsOut[0].w)
	}

	if n2.weightsIn[0].w != 0.5 {
		t.Errorf("expected weight from n1 to n2 to be 0.5 but got %f", n2.weightsOut[0].w)
	}
}
