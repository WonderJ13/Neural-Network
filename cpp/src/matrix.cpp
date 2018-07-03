#include <iostream>
#include <stdlib.h>
#include "matrix.h"

Matrix::Matrix(int rows_, int cols_)
{
	create_matrix(rows_, cols_);
}

Matrix::~Matrix()
{
	delete_matrix();
}

void Matrix::create_matrix(int rows_, int cols_) {
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

void Matrix::delete_matrix() {
	for(int i = 0; i < rows; i++) {
		delete [] matrix[i];
		matrix[i] = NULL;
	}
	delete [] matrix;
	matrix = NULL;
}

void Matrix::printMatrix()
{
	std::cout << "Rows: " << rows << "&" << getRows() << std::endl;
	std::cout << "Cols: " << cols << "&" << getCols() << std::endl;
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
		for(int j = 0; j < cols; j++) { //Makes a value between -1.0 and 1.0
			matrix[i][j] = (((double)rand() / RAND_MAX) * 2.0) - 1.0;
		}
	}
}

double Matrix::getNum(int i, int j) {
	return matrix[i][j];
}

void Matrix::setNum(double num, int i, int j) {
	matrix[i][j] = num;
}

Matrix* Matrix::add(Matrix* a, Matrix* b) {
	if(a->getRows() != b->getRows() || a->getCols() != b->getCols()) {
		std::cerr << "add error: " << a->getRows() << "x" << a->getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* sum = new Matrix(a->getRows(), a->getCols());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < a->getCols(); j++) {
			sum->setNum(a->getNum(i, j) + b->getNum(i, j), i, j);
		}
	}
	return sum;
}

Matrix* Matrix::subtract(Matrix* a, Matrix* b) {
	if(a->getRows() != b->getRows() || a->getCols() != b->getCols()) {
		std::cerr << "subtract error: " << a->getRows() << "x" << a->getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* diff = new Matrix(a->getRows(), a->getCols());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < a->getCols(); j++) {
			diff->setNum(a->getNum(i, j) - b->getNum(i, j), i, j);
		}
	}
	return diff;
}

Matrix* Matrix::multiply(Matrix* a, Matrix* b) {
	if(a->getRows() != b->getRows() || a->getCols() != b->getCols()) {
		std::cerr << "hadaman_multiply error: " << a->getRows() << "x" << a->getCols() << " doesn't match up with " << b->getRows() << "x" << b->getCols() << std::endl;
		throw 1;
	}
	Matrix* product = new Matrix(a->getRows(), a->getCols());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < a->getCols(); j++) {
			product->setNum(a->getNum(i, j) * b->getNum(i, j), i, j);
		}
	}
	return product;
}

Matrix* Matrix::multiply(Matrix* a, double n) {
	Matrix* product = new Matrix(a->getRows(), a->getCols());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < a->getCols(); j++) {
			product->setNum(a->getNum(i, j) * n, i, j);
		}
	}
	return product;
}

Matrix* Matrix::matrix_multiply(Matrix* a, Matrix* b) {
	if(a->getCols() != b->getRows()) {
		std::cerr << "matrix_multiply error: " << a->getCols() << " and " << b->getRows() << " don't match." << std::endl;
		throw 1;
	}
	Matrix* product = new Matrix(a->getRows(), b->getCols());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < b->getCols(); j++) {
			double sum = 0.0;
			for(int k = 0; k < a->getCols(); k++) {
				sum += a->getNum(i, k) * b->getNum(k, j);
			}
			product->setNum(sum, i, j);
		}
	}
	return product;
}

Matrix* Matrix::transpose(Matrix* a) {
	Matrix* ret = new Matrix(a->getCols(), a->getRows());
	for(int i = 0; i < a->getRows(); i++) {
		for(int j = 0; j < a->getCols(); j++) {
			ret->setNum(a->getNum(i, j), j, i);
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

Matrix* Matrix::copy() {
	Matrix* ret = new Matrix(rows, cols);
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
			ret->setNum(getNum(i, j), i, j);
		}
	}
	return ret;
}