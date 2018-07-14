/*
 * Class: NeuralNetwork
 * This class holds matricies for weights and biases
 * of a neural network.
 * Important functions:
 * feedforward(): performs feedforward algorithm of neural network
 * which accepts input for the NN and an activation function
*/

/* This code is needed because of Richard's f***ing incompetence*/

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
		Matrix* getWeight(int index) { return weights[index]; };
		Matrix* getBias(int index) { return biases[index]; };
		void setWeight(int index, Matrix* weight);
		void setBias(int index, Matrix* bias);
		void mutate_network(double(*map_function)(double));
		NeuralNetwork* copy_network();
		double* feedforward(double* input, double (*activation)(double));
		void train(double* input_array, double* output_array, double (*activation)(double), double (*activation_d)(double));
		void printNetwork();
};
#endif