
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

	__shared__ double At_s[TILE_WIDTH][TILE_WIDTH];
	__shared__ double A_s[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // symmetrc matrix padding
	if (blockIdx.x < blockIdx.y) return;

	double Pvalue;
	Pvalue = 0.0;
	int m;
	int loop;
	int mod = rows % TILE_WIDTH;

	// ittaration for N-1 tiling steps
	if (mod > 0) loop = (rows / TILE_WIDTH);
	else loop = (rows / TILE_WIDTH) - 1;
	//printf("Loop %d", loop);
	for (m = 0; m < loop; m++) {

		//initializing the shared memory matrces
		if (Row < rows) {
			At_s[ty][tx] = device_a[(m * TILE_WIDTH + tx) * columns + Row];
		}
		else {
			At_s[ty][tx] = 0;
		}

		if (Col < columns) {
			A_s[ty][tx] = device_a[(m * TILE_WIDTH + ty) * columns + Col];
		}
		else {
			A_s[ty][tx] = 0;
		}

		__syncthreads();

		//calculating the not-final results
		for (int k = 0; k < TILE_WIDTH; k++) {
			Pvalue += At_s[ty][k] * A_s[k][tx];
		}

		__syncthreads();

	}

	//The last step of tiling (special treatment)
	int remaining_tile_length = rows - m * TILE_WIDTH;
	if (ty >= remaining_tile_length) {
		A_s[ty][tx] = 0;
	}
	else {
		A_s[ty][tx] = device_a[(m * TILE_WIDTH + ty) * columns + Col];
	}

	if (tx >= remaining_tile_length) {
		At_s[ty][tx] = 0;
	}
	else {
		At_s[ty][tx] = device_a[(m * TILE_WIDTH + tx) * columns + Row];
	}

	__syncthreads();

	//final results calculation
	for (int k = 0; k < remaining_tile_length; k++) {
		Pvalue += At_s[ty][k] * A_s[k][tx];
	}

	// transfering results to global memory
	if (Row * columns + Col < columns * rows && Col * columns + Row < columns * rows) {
		device_c[Row * columns + Col] = Pvalue;
		device_c[Col * columns + Row] = Pvalue;
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
	int rows = atoi(argv[1]), columns = atoi(argv[2]);
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
	unsigned int numBlocksX = ((double)(columns - 1) / BLOCK_SIZE_PER_DIM + 1);
	unsigned int numBlocksY = ((double)(rows - 1) / BLOCK_SIZE_PER_DIM + 1);
	dim3 dimGrid(numBlocksX, numBlocksY);
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM);
	//-------------------------------------------------


	// save the A array for checking if result is in line
	/*ofstream input_stream;
	input_stream.open("input2.txt");
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < columns; c++) {
			input_stream << host_a[r * columns + c] << "\t";
		}
		input_stream << endl;
	}
	input_stream.close();*/


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

	//Save output file for testing
	/*ofstream output_stream;
	output_stream.open("outputer3.txt");
	for (int r = 0; r < columns; r++) {
		for (int c = 0; c < columns; c++) {
			output_stream << host_c[r * columns + c] << "\t";
		}
		output_stream << endl;
	}
	output_stream.close();*/

	// Free alocated space for arrays
	cudaFree(dev_a);
	cudaFree(dev_c);
	free(host_a);
	free(host_c);
	//--------------------------------

	return 0;
}
