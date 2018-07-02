#include <iostream>
#include <stack>
#include <cmath>
#include "neuralnetwork.h"

int gameDone(char**);
void drawBoard(char**);
bool makeMove(int, int, char**);
bool boardFull(char**);
double* convertToArray(char**, int);
double sigmoid(double);
double sigmoid_d(double);

int main() {
	int gameNumber = 0;
	int arr[4] = {42, 26, 13, 7};
	NeuralNetwork** nns = new NeuralNetwork*[2];
	nns[0] = new NeuralNetwork(arr, 4, 0.1);
	nns[1] = new NeuralNetwork(arr, 4, 0.1);
	while(1) {
		gameNumber++;
		//Setting up board
		char** board = new char*[6];
		for(int i = 0; i < 6; i++)
			board[i] = new char[7];
		for(int i = 0; i < 6; i++)
			for(int j = 0; j < 7; j++)
				board[i][j] = 0;

		//Gameplay
		std::stack<int> p1Moves;
		std::stack<double*> p1Boards, p1Outputs;
		std::stack<int> p2Moves;
		std::stack<double*> p2Boards, p2Outputs;
		while(!gameDone(board) && !boardFull(board)) { //Check after player 2
			double max; //Setting up variables for making decision
			int index;

			//Player 1 Start
			double* p1In = convertToArray(board, 1);
			double* p1Out = nns[0]->feedforward(p1In, sigmoid);

			p1Boards.push(p1In);
			p1Outputs.push(p1Out);

			//Search for move NN will take
			max = 0.0;
			index = 0;
			for(int i = 0; i < 7; i++) {
				if(max < p1Out[i]) {
					max = p1Out[i];
					index = i;
				}
			}

			//This only happens when the move picked can't be played
			while(!makeMove(1, index, board)) { //Look for next highest probability for move
				double unusableMax = max;
				max = 0.0;
				index = 0;
				for(int i = 0; i < 7; i++) {
					if(p1Out[i] >= unusableMax) continue;
					if(max < p1Out[i]) {
						max = p1Out[i];
						index = i;
					}
				}
			}

			p1Moves.push(index); //Remember moves for training

			if(gameNumber % 10000 == 0) drawBoard(board);
			if(gameDone(board)) break; //Check for win after player 1 makes a move

			/*int _;
			std::cin >> _;//Enter for next move*/

			//Player 2 Start
			double* p2In = convertToArray(board, 2);
			double* p2Out = nns[1]->feedforward(p2In, sigmoid);

			p2Boards.push(p2In);
			p2Outputs.push(p2Out);

			//Search for move NN will make
			max = 0.0;
			index = 0;
			for(int i = 0; i < 7; i++) {
				if(max < p2Out[i]) {
					max = p2Out[i];
					index = i;
				}
			}

			//This only happens when the move picked can't be played
			while(!makeMove(2, index, board)) { //Look for next highest probability for move
				double unusableMax = max;
				max = 0;
				index = 0;
				for(int i = 0; i < 7; i++) {
					if(p2Out[i] >= unusableMax) continue;
					if(max < p2Out[i]) {
						max = p2Out[i];
						index = i;
					}
				}
			}

			p2Moves.push(index); //Remember moves for training

			if(gameNumber % 10000 == 0) drawBoard(board);
			//std::cin >> _; //enter for next move
		}

		//Game end
		int winner = gameDone(board);
		//drawBoard(board);

		int p1Msize = p1Moves.size();
		int p2Msize = p2Moves.size();

		bool stop = false;
		if(winner == 0) {
			stop = true;
		} else if(winner == 1) {
			for(int i = 0; i < p1Msize; i++) {
				double* p1Board = p1Boards.top();
				double* p1Output = p1Outputs.top();
				int p1Index = p1Moves.top();
				p1Boards.pop();
				p1Outputs.pop();
				p1Moves.pop();

				p1Output[p1Index] = 1.;
				nns[0]->train(p1Board, p1Output, sigmoid, sigmoid_d);
				delete [] p1Board;
				delete [] p1Output;
			}
			for(int i = 0; i < p2Msize; i++) {
				double* p2Board = p2Boards.top();
				double* p2Output = p2Outputs.top();
				int p2Index = p2Moves.top();
				p2Boards.pop();
				p2Outputs.pop();
				p2Moves.pop();

				p2Output[p2Index] = 0.;
				nns[1]->train(p2Board, p2Output, sigmoid, sigmoid_d);
				delete [] p2Board;
				delete [] p2Output;
			}
		} else {
			for(int i = 0; i < p1Msize; i++) {
				double* p1Board = p1Boards.top();
				double* p1Output = p1Outputs.top();
				int p1Index = p1Moves.top();
				p1Boards.pop();
				p1Outputs.pop();
				p1Moves.pop();

				p1Output[p1Index] = 0.;
				nns[0]->train(p1Board, p1Output, sigmoid, sigmoid_d);
				delete [] p1Board;
				delete [] p1Output;
			}
			for(int i = 0; i < p2Msize; i++) {
				double* p2Board = p2Boards.top();
				double* p2Output = p2Outputs.top();
				int p2Index = p2Moves.top();
				p2Boards.pop();
				p2Outputs.pop();
				p2Moves.pop();

				p2Output[p2Index] = 1.;
				nns[1]->train(p2Board, p2Output, sigmoid, sigmoid_d);
				delete [] p2Board;
				delete [] p2Output;
			}
		}
		//if(gameNumber % 1000 == 0) std::cout << gameNumber << std::endl;

		//Delete Board
		for(int i = 0; i < 6; i++) {
			delete [] board[i];
			board[i] = NULL;
		}
		delete [] board;
		board = NULL;

		/*if(!winner) std::cout << "It's a draw!" << std::endl;
		else std::cout << "Player " << winner << " wins!" << std::endl;
		std::cout << "Play again?\n>>>";*/
		if(stop) break;
	}
	std::cout << "DONE!" << std::endl;
}

int gameDone(char** board) {
	//Rows Check
	for(int y = 0; y < 6; y++) {
		for(int x = 0; x < 4; x++) {
			if(!board[y][x]) continue;
			if(board[y][x] == board[y][x+1] && board[y][x] == board[y][x+2] && board[y][x] == board[y][x+3]) return board[y][x];
		}
	}
	//Cols Check
	for(int y = 0; y < 3; y++) {
		for(int x = 0; x < 7; x++) {
			if(!board[y][x]) continue;
			if(board[y][x] == board[y+1][x]  && board[y][x] == board[y+2][x] && board[y][x] == board[y+3][x]) return board[y][x];
		}
	}
	//Diag-Down Check
	for(int y = 0; y < 3; y++) {
		for(int x = 0; x < 4; x++) {
			if(!board[y][x]) continue;
			if(board[y][x] == board[y+1][x+1] && board[y][x] == board[y+2][x+2] && board[y][x] == board[y+3][x+3]) return board[y][x];
		}
	}
	//Diag-Up Check
	for(int y = 0; y < 3; y++) {
		for(int x = 6; x > 2; x--) {
			if(!board[y][x]) continue;
			if(board[y][x] == board[y+1][x-1] && board[y][x] == board[y+2][x-2] && board[y][x] == board[y+3][x-3]) return board[y][x];
		}
	}
	return 0; //game may not be finished yet
}

void drawBoard(char** board) {
	for(int y = 0; y < 6; y++) {
		for(int x = 0; x < 7; x++) {
			std::cout << (int)board[y][x];
			if(x < 6) std::cout << "|";
		}
		std::cout << std::endl;
	}
	std::cout << "----------------------------------" << std::endl;
}

bool makeMove(int player, int col, char** board) {
	int y = 5;
	while(y >= 0) {
		if(!board[y][col]) {
			board[y][col] = player;
			return true;
		}
		y--;
	}
	return false;
}

bool boardFull(char** board) {
	for(int y = 0; y < 6; y++) {
		for(int x = 0; x < 7; x++) {
			if(!board[y][x]) return false;
		}
	}
	return true;
}

double* convertToArray(char** board, int player) {
	double* ret = new double[42];
	for(int y = 0; y < 6; y++) {
		for(int x = 0; x < 7; x++) {
			if(board[y][x] == player) ret[y*7+x] = 1.; //NN will read spot with it's color as 1.0
			else if(board[y][x] == 0) ret[y*7+x] = 0.5; //NN will read free spot with 0.5
			else ret[y*7+x] = 0.0; //NN will read spot with opponent's color as 0.0
		}
	}
	return ret;
}

double sigmoid(double n) {
	double e2n = exp(-n);
	return 1 / (1 + e2n);
}

double sigmoid_d(double s_out) {
	return s_out * (1 - s_out);
}