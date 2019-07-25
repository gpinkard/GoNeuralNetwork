package main

import (
	"GoNeuralNetwork/Network"
	"fmt"
)

/*
for basic testing purposes atm
*/

func main() {
	fmt.Println("GNN")
	trainingData := [][]float64{{0.22323, 0.49734, 0.10112}}
	network := GoNeuralNetwork.CreateNetwork()
	network.AddLayer("basiclayer", "sigmoid", 3)
	network.AddLayer("basiclayer", "sigmoid", 3)
	network.AddLayer("basiclayer", "sigmoid", 3)
	network.AddLayer("basiclayer", "sigmoid", 3)
	output, _ := network.CalcNetworkOutput(trainingData[0])
	fmt.Println("output:", output)
	network.Train(trainingData, 0.3, 10)
}
