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

void NeuralNetwork::setWeight(int index, Matrix* weight) {
	delete weights[index];
	weights[index] = weight;
}

void NeuralNetwork::setBias(int index, Matrix* bias) {
	delete biases[index];
	biases[index] = bias;
}

void NeuralNetwork::mutate_network(double(*map_function)(double)) {
	for(int i = 0; i < layers-1; i++) {
		weights[i]->map(map_function);
		biases[i]->map(map_function);
	}
}

NeuralNetwork* NeuralNetwork::copy_network() {
	int arr[layers];
	for(int i = 0; i < layers; i++) {
		arr[i] = 1;
	}
	NeuralNetwork* ret = new NeuralNetwork(arr, layers, learning_rate);
	for(int i = 0; i < layers-1; i++) {
		//Copy each weight and bias, so this NN won't change
		//won't change As the other gets trained.
		ret->setWeight(i, weights[i]->copy());
		ret->setBias(i, biases[i]->copy());
	}
	return ret;
}

double* NeuralNetwork::feedforward(double* input, double (*activation)(double)) {
	int size = weights[0]->getCols();
	Matrix* current_layer = new Matrix(size, 1);
	for(int i = 0; i < size; i++) { //Make matrix to act as current layer
		current_layer->setNum(input[i], i, 0);
	}
	for(int i = 0; i < layers-1; i++) { //Feedforward algorithm
		Matrix* tmp = Matrix::matrix_multiply(weights[i], current_layer);
		Matrix* next_layer = Matrix::add(tmp, biases[i]);
		delete current_layer;
		delete tmp;
		tmp = NULL;

		next_layer->map(activation);
		current_layer = next_layer;
		next_layer = NULL;
	}
	double* ret = new double[current_layer->getRows()];
	for(int i = 0; i < current_layer->getRows(); i++) { //Organize output into 1-dimensional array
		ret[i] = current_layer->getNum(i, 0);
	}

	delete current_layer; //Delete output-layer
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
		Matrix* tmp = Matrix::matrix_multiply(weights[i-1], node_layers[i-1]);
		Matrix* next_layer = Matrix::add(tmp, biases[i-1]);
		delete tmp;
		tmp = NULL;

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
		Matrix* tmp = Matrix::multiply(node_layers[i+1], errors);
		Matrix* gradient = Matrix::multiply(tmp, learning_rate);
		delete tmp;
		delete node_layers[i+1];
		node_layers[i+1] = gradient;
		gradient = NULL;
		tmp = NULL;

		Matrix* prev_layer_T = Matrix::transpose(node_layers[i]);
		Matrix* weight_deltas = Matrix::matrix_multiply(node_layers[i+1], prev_layer_T);

		Matrix* new_weights = Matrix::add(weights[i], weight_deltas);
		Matrix* new_bias = Matrix::add(biases[i], node_layers[i+1]);
		delete weights[i];
		delete biases[i];

		weights[i] = new_weights;
		biases[i] = new_bias;

		if(i != 0) {
			//Calculate errors for previous layer
			//Skip if we're on last layer
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

void NeuralNetwork::printNetwork() {
	for(int i = 0; i < layers-1; i++) {
		std::cout << "Weight index " << i << std::endl;
		weights[i]->printMatrix();
		std::cout << "Bias index " << i << std::endl;
		biases[i]->printMatrix();
	}
	std::cout << std::endl;
}