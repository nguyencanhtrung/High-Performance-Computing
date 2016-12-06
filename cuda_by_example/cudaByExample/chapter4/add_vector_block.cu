/*******************************************************************
* Trung C. Nguyen - TU Kaiserslautern 
* Parallel implementation - using parallel blocks
********************************************************************/
#include "../common/book.h"
#include <stdio.h>
#include <stdlib.h> 			// rand()

#define NUM_BLOCKS 	10000

__global__ void add_vector(int *a, int *b, int *c, int vector_size){
	int tid = blockIdx.x;
	while(tid < vector_size){
		c[tid] 	= a[tid] + b[tid];
		tid	+= gridDim.x;
	}
}

int main(void){
	int vector_size = 460000000;
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;
	int *gpu_result;

	// Allocate host memory
	a = (int *)malloc( vector_size * sizeof(int));
	b = (int *)malloc( vector_size * sizeof(int));
	c = (int *)malloc( vector_size * sizeof(int));
	gpu_result = (int *)malloc( vector_size * sizeof(int));
	if( (a == NULL) || (b == NULL) || (c == NULL) || (gpu_result == NULL) ){
		return -1;
	}
	// Allocate device memory
	HANDLE_ERROR(cudaMalloc( (void**)&dev_a, vector_size * sizeof(int) ) );
	HANDLE_ERROR(cudaMalloc( (void**)&dev_b, vector_size * sizeof(int) ) );
	HANDLE_ERROR(cudaMalloc( (void**)&dev_c, vector_size * sizeof(int) ) );

	// Fill host memory
	for(int i = 0; i < vector_size; i++){
		a[i] = i + 100 * rand();
		b[i] = i + 10 * rand();
	}

	// Compute on host
	for(int i = 0; i < vector_size; i++){
		c[i] = a[i] + b[i];
	}

	// Pass input to device memory
	HANDLE_ERROR( cudaMemcpy( dev_a, a, vector_size * sizeof(int), cudaMemcpyHostToDevice ) ); 
	HANDLE_ERROR( cudaMemcpy( dev_b, b, vector_size * sizeof(int), cudaMemcpyHostToDevice ) ); 

	// Compute on GPU
	add_vector<<<NUM_BLOCKS,1>>>(dev_a, dev_b, dev_c, vector_size);

	// Sending final result to host
	HANDLE_ERROR( cudaMemcpy( gpu_result, dev_c, vector_size * sizeof(int), cudaMemcpyDeviceToHost ) ); 
	
	// Compare results
	int errorCount = 0;
	for(int i = 0; i < vector_size; i++){
		if(c[i] != gpu_result[i]) {
			++errorCount;
		}
	}

	if(errorCount != 0) {
		printf("ERROR !\n");
	} else {
		printf("SUCCESS! !\n");
	}

	// Free Device Memory
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	return 0;
}
