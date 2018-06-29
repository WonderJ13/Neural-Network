#include "neuralnetwork.h"
#include "matrix.h"
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <time.h>

double sigmoid(double n);
double sigmoid_d(double s_out);
double times2(double n);

int main()
{
	//srand(time(NULL));
	int rounds = 1000000;
	int arr[3] = {2, 2, 1};
	NeuralNetwork* nn = new NeuralNetwork(arr, 3, 0.1);

	double training_data[4][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
	double expected_result[4][1] = {{0.}, {1.}, {1.}, {0.}};
	for(int i = 0; i < rounds; i++) {
		int rnd = rand() % 4;
		nn->train(training_data[rnd], expected_result[rnd], sigmoid, sigmoid_d);
	}
	std::cout << "Output for {0, 0}: " << nn->feedforward(training_data[0], sigmoid)[0] << std::endl;
	std::cout << "Output for {0, 1}: " << nn->feedforward(training_data[1], sigmoid)[0] << std::endl;
	std::cout << "Output for {1, 0}: " << nn->feedforward(training_data[2], sigmoid)[0] << std::endl;
	std::cout << "Output for {1, 1}: " << nn->feedforward(training_data[3], sigmoid)[0] << std::endl << std::endl;
	delete nn;
}

double sigmoid(double n) {
	double e2n = exp(-n);
	return 1 / (1 + e2n);
}

double sigmoid_d(double s_out) {
	return s_out * (1 - s_out);
}

double times2(double n) {
	return n*2;
}