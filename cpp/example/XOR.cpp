#include "../src/neuralnetwork.h"
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <time.h>

double sigmoid(double n);
double sigmoid_d(double s_out);

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
	double* out00 = nn->feedforward(training_data[0], sigmoid);
	double* out01 = nn->feedforward(training_data[1], sigmoid);
	double* out10 = nn->feedforward(training_data[2], sigmoid);
	double* out11 = nn->feedforward(training_data[3], sigmoid);
	std::cout << "Output for {0, 0}: " << out00[0] << std::endl;
	std::cout << "Output for {0, 1}: " << out01[0] << std::endl;
	std::cout << "Output for {1, 0}: " << out10[0] << std::endl;
	std::cout << "Output for {1, 1}: " << out11[0] << std::endl << std::endl;
	delete out00;
	delete out01;
	delete out10;
	delete out11;
	delete nn;
}

double sigmoid(double n) {
	double e2n = exp(-n);
	return 1 / (1 + e2n);
}

double sigmoid_d(double s_out) {
	return s_out * (1 - s_out);
}