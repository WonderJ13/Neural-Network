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
	Matrix* current_layer = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Make matrix for to act as current layer
		current_layer->setNum(input[i], i, 0);
	}
	for(int i = 0; i < layers-1; i++) { //Feedforward algorithm
		Matrix* temp = Matrix::matrix_multiply(weights[i], current_layer);
		temp->add(biases[i]);
		temp->map(activation);
		delete current_layer;
		current_layer = temp;
	}
	double* ret = new double[current_layer->getRows()];
	for(int i = 0; i < current_layer->getRows(); i++) { //Organize output into a 1-dimensional array
		ret[i] = current_layer->getNum(i, 0);
	}
	delete current_layer;
	current_layer = NULL;
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
		Matrix* next_layer = Matrix::matrix_multiply(weights[i-1], node_layers[i-1]);
		next_layer->add(biases[i-1]);
		next_layer->map(activation);
		node_layers[i] = next_layer;
		next_layer = NULL;
	}

	size = node_layers[layers-1]->getRows();
	Matrix* target = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Copy values of target pointer to a Matrix
		target->setNum(output_array[i], i, 0);
	}
	Matrix* errors = Matrix::subtract(target, node_layers[layers-1]); //Get output errors

	for(int i = layers-2; i >= 0; i--) { //Update each weight according to errors of current layer
		node_layers[i+1]->map(activation_d);
		Matrix* gradient = Matrix::multiply(node_layers[i+1], errors);
		gradient->multiply(learning_rate);
		delete node_layers[i+1];
		node_layers[i+1] = gradient;
		gradient = NULL;

		Matrix* prev_layer_T = Matrix::transpose(node_layers[i]);
		Matrix* weight_deltas = Matrix::matrix_multiply(node_layers[i+1], prev_layer_T);

		weights[i]->add(weight_deltas);
		biases[i]->add(node_layers[i+1]);

		if(i != 0) { //Calculate errors for previous layer, not needed if we're at the beginning
			Matrix* weight_T = Matrix::transpose(weights[i]);
			Matrix* newerrors = Matrix::matrix_multiply(weight_T, errors);
			delete errors;
			delete weight_T;
			errors = newerrors;
			newerrors = NULL;
			weight_T = NULL;
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