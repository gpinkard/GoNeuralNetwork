package GoNeuralNetwork

import (
	"errors"
	"math"
)

/*
An interface representing an activation function.
*/
type actfunc interface {
	// calculate output of interface given a weighted sum
	calculate(float64) float64
	// calculate delta value for a neuron
	calculateDelta(float64, float64, float64, float64, bool) float64
	// return string representation of this actfunc (i.e. "sigmoid")
	getActFunc() string
}

type sigmoid struct{}

type relu struct{}

/*
Creates and returns an activation function, given a string representing the activation
function. Returns a non-nil error if the given string does not correspond to a valid
activation function.
*/
func createActFunc(actfuncType string) (actfunc, error) {
	switch actfuncType {
	case "sigmoid":
		return sigmoid{}, nil
	case "relu":
		return relu{}, nil
	default:
		return nil, errors.New(actfuncType + " is not a valid activation function")
	}
}

/*
Calculates output of sigmoid.
*/
func (s sigmoid) calculate(weightedSum float64) float64 {
	return 1.0 / (1.0 + (math.Pow(math.E, (weightedSum * -1.0))))
}

/*
Calculates delta value for sigmoid.
*/
func (s sigmoid) calculateDelta(weight, output, trainingDatum, prevDelta float64, isHiddenLayer bool) float64 {
	if isHiddenLayer {
		return output * (1 - output) * weight * prevDelta
	}
	return output * (1 - output) * (trainingDatum - output)
}

/*
Returns type of actfunc ("sigmoid").
*/
func (s sigmoid) getActFunc() string {
	return "sigmoid"
}

/*
Calculates output of ReLU.
*/
func (r relu) calculate(weightedSum float64) float64 {
	if weightedSum > 0.0 {
		return weightedSum
	}
	return 0.0
}

/*
Calculates delta value of ReLU.
*/
func (r relu) calculateDelta(weight, output, trainingDatum, prevDelta float64, isHiddenLayer bool) float64 {
	return 0.0 // TODO
}

/*
Retuns actfunc type ("relu").
*/
func (r relu) getActFunc() string {
	return "relu"
}
