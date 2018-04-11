/*
 * Class: NeuralNetwork
 * This class holds matricies for weights and biases
 * of a neural network.
 * Important functions:
 * feedforward(): performs feedforward algorithm of neural network
 * with an input corresponding to the input layer and an input activation function
*/

#ifndef NN_H
#define NN_H
#include "matrix.h"

class NeuralNetwork {
	private:
		Matrix** weights;
		Matrix** biases;
		int layers;
		double learning_rate;

	public:
		NeuralNetwork(int neuron_count[], int size, double learning_rate_);
		~NeuralNetwork();
		double* feedforward(double* input, double (*activation)(double));
		void train(double* input_array, double* output_array, double (*activation)(double), double (*activation_d)(double));
		Matrix** getWeights() { return weights; };
};
#endif