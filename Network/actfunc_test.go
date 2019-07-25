package GoNeuralNetwork

import (
	"math"
	"testing"
)

/*
Tests for actfunc.go
*/

func TestCreateActFunc(t *testing.T) {
	_, err := createActFunc("notanactfunc")
	if err == nil {
		t.Errorf("Exprected error to be returned, but got none")
	}
}

func TestGetActFunc(t *testing.T) {
	sigmoid, _ := createActFunc("sigmoid")
	if sigmoid.getActFunc() != "sigmoid" {
		t.Errorf("Expected activation function to be sigmoid but got " + sigmoid.getActFunc())
	}

	relu, _ := createActFunc("relu")
	if relu.getActFunc() != "relu" {
		t.Errorf("Expected activation function to be relu but got " + sigmoid.getActFunc())
	}
}

func TestCalculate(t *testing.T) {
	const TOLERANCE = 0.000001

	sigmoid, _ := createActFunc("sigmoid")
	res := math.Abs(sigmoid.calculate(0.5) - 0.622459)
	if res > TOLERANCE {
		t.Errorf("Expected value of 0.622459, but got %f", res)
	}

	relu, _ := createActFunc("relu")
	res = relu.calculate(0.0)
	if res != 0.0 {
		t.Errorf("Expected value of 0.0, but got %f", res)
	}
	res = relu.calculate(0.3)
	if res != 0.3 {
		t.Errorf("Expected value of 0.3, but got %f", res)
	}
}

func TestCalculateDelta(t *testing.T) {
	sigmoid, _ := createActFunc("sigmoid")
	res := sigmoid.calculateDelta(0.5, 0.5, 0.5, 0.5, true)
	if res != 0.0625 {
		t.Errorf("Expected value of 0.0625, but got %f", res)
	}

	res = sigmoid.calculateDelta(0.5, 0.5, 0.5, 0.5, false)
	if res != 0.0 {
		t.Errorf("Expected value of 0.0, but got %f", res)
	}

	// TODO finish relu and then write tests
}
