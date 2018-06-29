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
	system("pause");
	/*system("pause");
	Matrix** hi = (Matrix**)malloc(sizeof(Matrix*)*500000);
	std::cout << sizeof(Matrix*) << std::endl;
	system("pause");
	for(int i = 0; i < 500000; i++) {
		//std::cout << i << std::endl;
		hi[i] = new Matrix(2, 2);
	}
	system("pause");
	free(hi);
	system("pause");*/
	/*Matrix* bias_h = new Matrix(2, 1);
	bias_h->setNum(-0.13712031503302669, 0, 0);
	bias_h->setNum(-0.365227866660748, 1, 0);
	//bias_h->printMatrix();

	Matrix* bias_o = new Matrix(1, 1);
	bias_o->setNum(-0.2099934948614357, 0, 0);
	//bias_o->printMatrix();

	Matrix* weight_ih = new Matrix(2, 2);
	weight_ih->setNum(-0.11457294114642247, 0, 0);
	weight_ih->setNum(0.10887493903866785, 0, 1);
	weight_ih->setNum(0.7760909382501211, 1, 0);
	weight_ih->setNum(0.29967160246951696, 1, 1);
	//weight_ih->printMatrix();

	Matrix* weight_ho = new Matrix(1, 2);
	weight_ho->setNum(0.3039753238002776, 0, 0);
	weight_ho->setNum(0.8383731718333678, 0, 1);
	//weight_ho->printMatrix();

	int nodes[] = {2, 2, 1};
	NeuralNetwork* nn = new NeuralNetwork(nodes, 3, 0.1);
	nn->setWeight(weight_ih, 0);
	nn->setWeight(weight_ho, 1);
	nn->setBias(bias_h, 0);
	nn->setBias(bias_o, 1);

	double in[] = {0, 1};
	double out[] = {1};

	double* output = nn->feedforward(in, sigmoid);
	std::cout << output[0] << std::endl;*/
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