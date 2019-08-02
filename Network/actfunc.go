package GoNeuralNetwork

import (
	"errors"
	"math"
)

type actfunc interface {
	calculate(float64) float64
	calculateDelta(float64, float64, float64, float64, bool) float64
	getActFunc() string
}

type sigmoid struct{}

type relu struct{}

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

func (s sigmoid) calculate(weightedSum float64) float64 {
	return 1.0 / (1.0 + (math.Pow(math.E, (weightedSum * -1.0))))
}

func (s sigmoid) calculateDelta(weight, output, trainingDatum, prevDelta float64, isHiddenLayer bool) float64 {
	if isHiddenLayer {
		return output * (1 - output) * weight * prevDelta
	}
	return output * (1 - output) * (trainingDatum - output)
}

func (s sigmoid) getActFunc() string {
	return "sigmoid"
}

func (r relu) calculate(weightedSum float64) float64 {
	if weightedSum > 0.0 {
		return weightedSum
	}
	return 0.0
}

func (r relu) calculateDelta(weight, output, trainingDatum, prevDelta float64, isHiddenLayer bool) float64 {
	return 0.0 // TODO
}

func (r relu) getActFunc() string {
	return "relu"
}
