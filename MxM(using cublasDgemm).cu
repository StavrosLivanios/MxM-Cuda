#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <cublas_v2.h>


using namespace std;

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
		//printf("x=%d - y=%d | C:%lf \n", Row,Col,Pvalue);
	}

}


void gpuCublas(cublasHandle_t& handle, const double* A, const double* B, double* C, const int m, const int k) {
	int lda = m, ldb = k, ldc = m;
	const double alf = 1;
	const double bet = 0;
	const double* alpha = &alf;
	const double* beta = &bet;
	cudaEvent_t start;
	cudaEvent_t stop;
	float kernel_time;
	// do the actual multiplication
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, m, k, alpha, A, lda, B, lda, beta, C, ldc);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&kernel_time, start, stop);

	cublasDestroy(handle);


	cout << "Time for kernel : " << kernel_time << endl;
}


int main(int argc, char* argv[])
{


	if (argc < 2) {
		fprintf(stderr, "Rerun with -.exe rows columns-\n");
		exit(1);
	}

	// variable initiation-----------------------------
	cudaDeviceProp prop;
	int BLOCK_SIZE_PER_DIM = 32;
	int rows = atoi(argv[1]);;
	int columns = atoi(argv[2]);;
	int Blocks_number;
	int size = rows * columns;
	int size_result = columns * columns;

	double* host_c, * host_a;//host matrixes
	double* dev_a, * dev_c; //device matrixes


	host_a = (double*)malloc(size * sizeof(double));
	host_c = (double*)malloc(size_result * sizeof(double));

	init(host_a, rows, columns);

	/*
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

	cudaError_t test = cudaGetDeviceProperties(&prop, 0);



	cudaMalloc((void**)&dev_c, size_result * sizeof(double));
	cudaMalloc((void**)&dev_a, size * sizeof(double));
	cudaMemcpy(dev_a, host_a, size * sizeof(double), cudaMemcpyHostToDevice);


	unsigned int numBlocksX = (columns - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (rows - 1) / BLOCK_SIZE_PER_DIM + 1;

	dim3 dimGrid(numBlocksX, numBlocksY, 1);
	dim3 dimBlock(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);

	const double h_one = 1; // constants used in
	const double h_zero = 0;
	cublasHandle_t handle;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublasCreate(&handle);

	gpuCublas(handle, dev_a, dev_a, dev_c, columns, rows);

	cudaMemcpy(host_c, dev_c, size_result * sizeof(double), cudaMemcpyDeviceToHost);

	/*
	ofstream output_stream;
	output_stream.open("outputer1.txt");
	for (int r = 0; r < columns; r++) {
		for (int c = 0; c < columns; c++) {
			output_stream << host_c[r*columns+c] << "\t";
		}
		output_stream <<endl;	}
	output_stream.close();
	*/

	cudaFree(dev_a);
	cudaFree(dev_c);
	free(host_a);
	free(host_c);


	return 0;
}
