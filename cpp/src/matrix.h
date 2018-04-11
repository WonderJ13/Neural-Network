/*
 * Class: Matrix
 * Class that holds multiple variables and treats them
 * as a mathematical matrix.
 * Important functions:
 * add(): adds two matricies together and returns the sum
 * matrix_multiply(): performs matrix multiplication of this and the input matrix
 * transpose(): performs a transpose of this matrix and returns it kepping this unchagned
*/
#ifndef MATRIX_H
#define MATRIX_H
class Matrix
{
	private:
		int rows, cols;
		double** matrix;
	public:
		Matrix(int rows, int cols);
		~Matrix();
		void printMatrix();
		void randomize();
		double getNum(int i, int j);
		void setNum(double num, int i, int j);
		Matrix* add(Matrix* b);
		Matrix* subtract(Matrix* b);
		Matrix* multiply(Matrix* b);
		Matrix* multiply(double n);
		Matrix* matrix_multiply(Matrix* b);
		Matrix* transpose();
		void map(double (*map_function)(double));
		double** getMatrix() { return matrix; };
		int getRows() { return rows; };
		int getCols() { return cols; };
};
#endif