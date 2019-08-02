package main

import (
	//"GoNeuralNetwork/Network"
	"GNN_TMP/Network"
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
	/*
		fmt.Println("")
		network.PrintNetwork()
		fmt.Println("")
	*/
	output, _ := network.CalculateNetworkOutput(trainingData[0])
	fmt.Println("output:", output)
	network.Train(trainingData, 0.3, 10)
	/*
		fmt.Println("")
		network.PrintNetwork()
		fmt.Println("")
	*/
}
