#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "neuralnetwork.h"

int gameDone(char**);
void drawBoard(char**);
bool makeMove(int, int, char**);
bool boardFull(char**);
double* convertToArray(char**, int);
double sigmoid(double);
double sigmoid_d(double);
double mutate(double);
double randomGaussian(double, double);

const int POPULATION = 100;
const double MUTATION_RATE = 0.1;

int main() {
	int generation = 0;
	int arr[4] = {42, 26, 13, 7};
	NeuralNetwork** nns = new NeuralNetwork*[POPULATION];
	for(int i = 0; i < POPULATION; i++) {
		nns[i] = new NeuralNetwork(arr, 4, 0.1);
	}
	while(1) {
		generation++;
		//std::cout << "Generation " << generation << std::endl;

		int* wins = new int[POPULATION];
		for(int i = 0; i < POPULATION; i++) {
			wins[i] = 0;
		}

		//Setting up competitions
		for(int p1 = 0; p1 < POPULATION; p1++) {
			for(int p2 = p1+1; p2 < POPULATION; p2++) {
				for(int side = 0; side < 2; side++) {
					char** board = new char*[6];
					for(int i = 0; i < 6; i++)
						board[i] = new char[7];
					for(int i = 0; i < 6; i++)
						for(int j = 0; j < 7; j++)
							board[i][j] = 0;

					int players_in_game[2] = {p1, p2}; //Players in game, [0] == player 1, [1] == player 2
					//Gameplay
					for(int player = 0; !gameDone(board) && !boardFull(board); (player + 1) % 2) { //Check after player 2
						double max; //Setting up variables for making decision
						int index;

						//Player 1 Start
						double* inputLayer = convertToArray(board, 1);
						double* outputLayer = nns[players_in_game[player]]->feedforward(p1In, sigmoid); //Get NN's move

						//Search for move NN will take
						max = 0.0;
						index = 0;
						for(int i = 0; i < 7; i++) {
							if(max < outputLayer[i]) {
								max = outputLayer[i];
								index = i;
							}
						}

						bool playercantPlay[7] = {false, false, false, false, false, false, false};

						//This only happens when the move picked can't be played
						while(!makeMove(1, index, board)) { //Look for next highest probability for move
							playercantPlay[index] = true;
							max = 0.0;
							index = 0;
							for(int i = 0; i < 7; i++) {
								if(playercantPlay[i]) continue;
								if(max < outputLayer[i]) {
									max = outputLayer[i];
									index = i;
								}
							}
						}

						delete [] inputLayer;
						delete [] outputLayer;
					}

					//Game end
					int winner = gameDone(board);
					//drawBoard(board);
					if(!winner) {
						wins[p1]++;
						wins[p2]++;
					} else if(winner == 1) {
						wins[p1] += 3;
					} else if(winner == 2) {
						wins[p2] += 3;
					}

					//Delete Board
					for(int i = 0; i < 6; i++) {
						delete [] board[i];
						board[i] = NULL;
					}
					delete [] board;
					board = NULL;

					int tmp = p1; //Reset game, have player 1 play second
					p1 = p2;	  //and player 2 play first
					p2 = tmp;	  //for fairness reasons
								  //This doesn't mess with 4loops, since this happens twice
				}
			}
		}
		int max_wins = 0;
		int index_of_best_nn = 0;
		int sum = 0;
		for(int i = 0; i < POPULATION; i++) {
			sum += wins[i];
			if(max_wins < wins[i]) {
				max_wins = wins[i];
				index_of_best_nn = i;
			}
		}
		if(generation % 100 == 0 || generation == 1) {
			std::cout << "Best network of generation " << generation << ":" << std::endl;
			nns[index_of_best_nn]->printNetwork();
			std::cout << std::endl << std::endl;
		}
		int population_selection[sum];
		int index = 0;
		for(int i = 0; i < POPULATION; i++) { //Create mating pool (those who won more are more prevalent in array)
			for(int j = 0; j < wins[i]; j++) {
				population_selection[index] = i;
				index++;
			}
		}
		NeuralNetwork** new_nns = new NeuralNetwork*[POPULATION];
		for(int i = 0; i < POPULATION; i++) { //Create child from random parent, mutate, and add it to next generation
			int parent = (int)(((double)rand() / RAND_MAX) * sum);
			while(parent == sum) parent = (int)(((double)rand() / RAND_MAX) * sum); //There is a chance parent can go out of bounds
			NeuralNetwork* child = nns[population_selection[parent]]->copy_network();
			child->mutate_network(mutate);
			new_nns[i] = child;
		}

		//Cleanup
		for(int i = 0; i < POPULATION; i++) {
			delete nns[i];
		}
		delete [] nns;
		delete [] wins;
		nns = new_nns; //Store children in set of players to face each other in the HOT GAME Connect 4
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
			if(board[y][x] == player) ret[y*7+x] = 0.5; //NN will read spot with it's color as 0.5
			else if(board[y][x] == 0) ret[y*7+x] = 1.0; //NN will read free spot with 1.0
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

double mutate(double in) {
	if(((double)rand() / RAND_MAX) < MUTATION_RATE) {
		return in + randomGaussian(0, 0.1);
	}
	return in;
}

double randomGaussian(double mean, double std) {
	double x1 = 0;
	double x2 = 0;
	double hyp = 0;
	do {
		x1 = (((double)rand() / RAND_MAX) * 2.0) - 1.0; //Random number between -1 && 1
		x2 = (((double)rand() / RAND_MAX) * 2.0) - 1.0; //Random number between -1 && 1
		hyp = x1*x1+x2*x2;
	} while(hyp >= 1);
	hyp = sqrt(-2 * log(hyp) / hyp);
	x1 = x1 * hyp;
	return x1 * std + mean;
}