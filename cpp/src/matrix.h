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
		void create_matrix(int rows, int cols);
		void delete_matrix();
	public:
		Matrix(int rows, int cols);
		~Matrix();
		void printMatrix();
		void randomize();
		double getNum(int i, int j);
		void setNum(double num, int i, int j);
		static Matrix* add(Matrix* a, Matrix* b);
		static Matrix* subtract(Matrix* a, Matrix* b);
		static Matrix* multiply(Matrix* a, Matrix* b);
		static Matrix* multiply(Matrix* a, double n);
		static Matrix* matrix_multiply(Matrix* a, Matrix* b);
		static Matrix* transpose(Matrix* a);
		void map(double (*map_function)(double));
		double** getMatrix() { return matrix; };
		int getRows() { return rows; };
		int getCols() { return cols; };
};
#endif