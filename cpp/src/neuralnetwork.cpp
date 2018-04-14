#include "neuralnetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork(int neuron_count[], int size, double learning_rate_) {
	layers = size;
	learning_rate = learning_rate_;
	weights = new Matrix*[layers-1];
	biases = new Matrix*[layers-1];
	for(int i = 1; i < layers; i++) { //Make weights and biases for each layer
		weights[i-1] = new Matrix(neuron_count[i], neuron_count[i-1]);
		weights[i-1]->randomize();
		biases[i-1] = new Matrix(neuron_count[i], 1);
		biases[i-1]->randomize();
	}
}

NeuralNetwork::~NeuralNetwork() {
	for(int i = 0; i < layers-1; i++) {
		delete weights[i];
		delete biases[i];
		weights[i] = NULL;
		biases[i] = NULL;
	}
	delete [] weights;
	delete [] biases;
	weights = NULL;
	biases = NULL;
}

double* NeuralNetwork::feedforward(double* input, double (*activation)(double)) {
	int size = weights[0]->getCols();
	Matrix* temp = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Make matrix for to act as current layer
		temp->setNum(input[i], i, 0);
	}
	for(int i = 0; i < layers-1; i++) { //Feedforward algorithm
		temp = weights[i]->matrix_multiply(temp);
		temp = temp->add(biases[i]);
		temp->map(activation);
	}
	double* ret = new double[temp->getRows()];
	for(int i = 0; i < temp->getRows(); i++) { //Organize output into a 1-dimensional array
		ret[i] = temp->getNum(i, 0);
	}
	delete temp;
	temp = NULL;
	return ret;
}

void NeuralNetwork::train(double* input_array, double* output_array, double (*activation)(double), double (*activation_d)(double)) {
	Matrix** node_layers = new Matrix*[layers];
	int size = weights[0]->getCols();
	node_layers[0] = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Copy values in pointer to a Matrix
		node_layers[0]->setNum(input_array[i], i, 0);
	}
	for(int i = 1; i < layers; i++) { //Feedforward while saving each layer
		node_layers[i] = weights[i-1]->matrix_multiply(node_layers[i-1]);
		node_layers[i] = node_layers[i]->add(biases[i-1]);
		node_layers[i]->map(activation);
	}

	size = node_layers[layers-1]->getRows();
	Matrix* target = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Copy values of target pointer to a Matrix
		target->setNum(output_array[i], i, 0);
	}
	Matrix* errors = target->subtract(node_layers[layers-1]); //Get output errors

	for(int i = layers-2; i >= 0; i--) { //Update each weight according to errors of current layer
		node_layers[i+1]->map(activation_d);
		node_layers[i+1] = node_layers[i+1]->multiply(errors);
		node_layers[i+1] = node_layers[i+1]->multiply(learning_rate);

		Matrix* prev_layer_T = node_layers[i]->transpose();
		Matrix* weight_deltas = node_layers[i+1]->matrix_multiply(prev_layer_T);

		weights[i] = weights[i]->add(weight_deltas);
		biases[i] = biases[i]->add(node_layers[i+1]);

		if(i != 0) { //Calculate errors for previous layer, not needed if we're at the beginning
			Matrix* weight_T = weights[i]->transpose();
			errors = weight_T->matrix_multiply(errors); //TODO: calculate errors before backpropogation
			delete weight_T;
		}
		delete prev_layer_T;
		delete weight_deltas;
		prev_layer_T = NULL;
		weight_deltas = NULL;
	}

	delete errors;
	delete target;
	errors = NULL;
	target = NULL;
	for(int i = 0; i < layers; i++) {
		delete node_layers[i];
		node_layers[i] = NULL;
	}
	delete [] node_layers;
	node_layers = NULL;
}