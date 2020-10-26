


#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <cublas_v2.h>
#include <device_functions.h>


using namespace std;
#define TILE_WIDTH 32


void init(double* A, int x, int y)
{
	srand(time(NULL));
	int i, j;

	for (i = 0; i < x; ++i) {
		for (j = 0; j < y; ++j) {
			A[i * y + j] = (double)(rand() % 100) + ((double)rand() / RAND_MAX);
		}
	}
}

void init_from_file(double* A, int x, int y)
{
	int i = 0, j = 0;

	ifstream file;
	file.open("input.txt");
	if (!file.is_open()) return;

	string word;
	while (file >> word)
	{
		A[i * y + j] = atof(word.c_str());
		j = j + 1;
		if (j % y == 0) {
			j = 0;
			i = i + 1;
		}
	}

}


//=-=-=-=-=-=-=-=-= Function of Kernel =-=-=-=-=-=-=-=-=-=-=-=-=-=-=

__global__ void MatrixMulKernel(double* device_a, double* device_c, int rows, int columns)
{
	// Calculate the row index of the P_d element and M_d
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the column idenx of P_d and N_d
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((Row < rows) && (Col < columns)) {

		double Pvalue = 0.0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < rows; k++)
			Pvalue += device_a[k * columns + Row] * device_a[k * columns + Col];

		device_c[Row * columns + Col] = Pvalue;
		//printf("x=%d - y=%d | C:%lf \n", Row, Col, Pvalue);
	}

}

//=-=-=-=-=-=-=-=-=-=-=-=-=-END OF KERNEL FUNCHION=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

int main(int argc, char* argv[])
{

	
	if (argc < 2) {
		fprintf(stderr, "Rerun with -.exe rows columns-\n");
		exit(1);
	}
	

	// variable initiation-----------------------------
	cudaEvent_t start, stop;
	cudaDeviceProp prop;
	float kernel_time;
	int BLOCK_SIZE_PER_DIM = 32;
	int rows = atoi(argv[1]);
	int columns = atoi(argv[2]);
	printf("x=%d - y=%d \n", rows, columns);
	int Blocks_number;
	int size = rows * columns;
	int size_result = columns * columns;
	double* host_c, * host_b, * host_a;//host matrixes b for cublass c for our's
	double* dev_c, * dev_b, * dev_a; //device matrixes
	//-------------------------------------------------

	cudaError_t test = cudaGetDeviceProperties(&prop, 0);

	// Array size allocation --------------------------
	host_a = (double*)malloc(size * sizeof(double));
	host_b = (double*)malloc(size_result * sizeof(double));
	host_c = (double*)malloc(size_result * sizeof(double));

	// initialize randomly the array A
	init(host_a, rows, columns);

	//relocate the arrays neede to the gpu global memory
	cudaMalloc((void**)&dev_c, size_result * sizeof(double));
	cudaMalloc((void**)&dev_a, size * sizeof(double));
	cudaMemcpy(dev_a, host_a, size * sizeof(double), cudaMemcpyHostToDevice);

	//Find the grid and block sizes
	unsigned int numBlocksX = (columns - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (rows - 1) / BLOCK_SIZE_PER_DIM + 1;
	dim3 dimGrid(numBlocksX, numBlocksY, 1);
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);
	//-------------------------------------------------


	/*
	// save the A array for checking if result is in line
	ofstream input_stream;
	input_stream.open("input2.txt");
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			input_stream << host_a[r * columns + c] << "\t";
		}
		input_stream << endl;
	}
	input_stream.close();
	*/

	//===============KERNEL===================
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	MatrixMulKernel << <  dimGrid, dimBlock >> > (dev_a, dev_c, rows, columns);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);
	//===============KERNEL END===================

	//print time it took to run and return result array to host
	cout << "Time for our kernel  : " << kernel_time << endl;
	cudaMemcpy(host_c, dev_c, size_result * sizeof(double), cudaMemcpyDeviceToHost);
	/*
	//Save output file for testing
	ofstream output_stream;
	output_stream.open("outputer2.txt");
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			output_stream << host_c[r * columns + c] << "\t";
		}
		output_stream << endl;
	}
	output_stream.close();
	*/

	// Free alocated space for arrays
	cudaFree(dev_a);
	cudaFree(dev_c);
	free(host_a);
	free(host_c);
	//--------------------------------

	return 0;
}



