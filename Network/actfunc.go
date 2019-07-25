package GoNeuralNetwork

import (
	"errors"
	"math"
	"strings"
)

/*
An interface for layer types for the neural network
*/
type actfunc interface {
	// calculate and return the output of this activation function given a weighted sum
	calculate(float64) float64
	// calculate and return the delta (error value) for a given neuron
	calculateDelta(float64, float64, float64, float64, bool) float64
	// returns the type of this actfunc
	getActFunc() string
}

type sigmoid struct {
	af string
}

type relu struct {
	af string
}

/*
Creates and returns an activation function specified by input string. Returns an error if string
is not a valid activation function
*/
func createActFunc(actfunc string) (actfunc, error) {
	actfunc = strings.ToLower(actfunc)
	switch actfunc {
	case "sigmoid":
		return sigmoid{
			af: "sigmoid",
		}, nil
	case "relu":
		return relu{
			af: "relu",
		}, nil
	}
	return nil, errors.New(actfunc + " is not a valid activation function")
}

/*
can add more activation functions as long as you include entry in createActFunc
switch statement
*/

/*
Calculates and returns output of sigmoid function
*/
func (s sigmoid) calculate(n float64) float64 {
	return 1.0 / (1.0 + (math.Pow(math.E, (n * -1.0))))
}

/*
Calculates and returns delta (error) for a sigmoid. Weight is the weight of the
connection, output is the output of the neuron we are calculating a delta value
for, trainingDatum (used by output layers only) is the value we expect the
neural network to output for this neuron, prevDelta (used only by hidden layers)
is the delta value of the neuron this neuron feeds into for this connection, and
isHiddenLayer is a boolean, true if this is to a hidden layer we are calculating
the delta for.
*/
func (s sigmoid) calculateDelta(weight, output, trainingDatum, prevDelta float64, isHiddenLayer bool) float64 {
	if isHiddenLayer {
		return output * (1 - output) * weight * prevDelta
	} else {
		return output * (1 - output) * (trainingDatum - output)
	}
}

/*
Retuns a string representing this activation function (literally "sigmoid")
*/
func (s sigmoid) getActFunc() string {
	return s.af
}

func (r relu) calculate(n float64) float64 {
	if n < 0.0 {
		return 0.0
	}
	return n
}

func (r relu) calculateDelta(weight, output, trainingDatum, prevDelta float64, hiddenLayer bool) float64 {
	// TODO
	return 0.0
}

func (r relu) getActFunc() string {
	return r.af
}
