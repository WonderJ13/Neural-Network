#include <iostream>
#include <stdlib.h>
#include "matrix.h"

Matrix::Matrix(int rows_, int cols_)
{
	rows = rows_;
	cols = cols_;
	matrix = new double*[rows];
	for(int i = 0; i < rows; i++) {
		matrix[i] = new double[cols];
	}
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			matrix[i][j] = 0;
		}
	}
}

Matrix::~Matrix()
{
	for(int i = 0; i < rows; i++) {
		delete [] matrix[i];
		matrix[i] = NULL;
	}
	delete [] matrix;
	matrix = NULL;
}

void Matrix::printMatrix()
{
	for(int i = 0; i < rows; i++) {
		std::cout << "[";
		for(int j = 0; j < cols; j++) {
			std::cout << matrix[i][j] << ", ";
		}
		std::cout << "\b\b]" << std::endl;
	}
	std::cout << std::endl;
}

void Matrix::randomize()
{
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) { //Makes a value between -1.0 and 1.0, there's probably a better way to do this but I'm lazy
			matrix[i][j] = (((double)rand() / (RAND_MAX + 1.)) - 0.5) * 2;
		}
	}
}

double Matrix::getNum(int i, int j) {
	return matrix[i][j];
}

void Matrix::setNum(double num, int i, int j) {
	matrix[i][j] = num;
}

Matrix* Matrix::add(Matrix* b) {
	if(rows != b->getRows() || cols != b->getCols()) {
		std::cerr << "add error: " << getRows() << "x" << getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* sum = new Matrix(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			sum->setNum(getNum(i, j) + b->getNum(i, j), i, j);
		}
	}
	return sum;
}

Matrix* Matrix::subtract(Matrix* b) {
	if(rows != b->getRows() || cols != b->getCols()) {
		std::cerr << "subtract error: " << getRows() << "x" << getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* diff = new Matrix(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			diff->setNum(getNum(i, j) - b->getNum(i, j), i, j);
		}
	}
	return diff;
}

Matrix* Matrix::multiply(Matrix* b) {
	if(rows != b->getRows() || cols != b->getCols()) {
		std::cerr << "subtract error: " << getRows() << "x" << getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* product = new Matrix(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			product->setNum(getNum(i, j) * b->getNum(i, j), i, j);
		}
	}
	return product;
}

Matrix* Matrix::multiply(double n) {
	Matrix* product = new Matrix(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			product->setNum(getNum(i, j) * n, i, j);
		}
	}
	return product;
}

Matrix* Matrix::matrix_multiply(Matrix* b) {
	if(getCols() != b->getRows()) {
		std::cerr << "matrix_multiply error: " << getCols() << " and " << b->getRows() << " don't match." << std::endl;
		throw 1;
	}
	Matrix* product = new Matrix(rows, b->cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < b->cols; j++) {
			for(int k = 0; k < cols; k++) {
				product->setNum(product->getNum(i, j) + getNum(i, k)*b->getNum(k, j), i, j);
			}
		}
	}
	return product;
}

Matrix* Matrix::transpose() {
	Matrix* ret = new Matrix(cols, rows);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			ret->setNum(matrix[i][j], j, i);
		}
	}
	return ret;
}

void Matrix::map(double (*map_function)(double)) {
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			matrix[i][j] = map_function(matrix[i][j]);
		}
	}
}