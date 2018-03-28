#include "utils.h"
#include <stdlib.h>

#include "summation_kernel.cu"

// CPU implementation
float log2_series(int n)
{
	float result = 0.0;
	for(int i = n-1; i >= 0; i--)
		result = i%2 ? result-1.0/(i+1) : result+1.0/(i+1);
	return result;
}

int main(int argc, char ** argv)
{
    int data_size = 1024 * 1024 * 128;

    // Run CPU version
    double start_time = getclock();
    float log2 = log2_series(data_size);
    double end_time = getclock();
    
    printf("CPU result: %f\n", log2);
    printf(" log(2)=%f\n", log(2.0));
    printf(" time=%fs\n", end_time - start_time);
    
    // Parameter definition
    const int threads_per_block = 4 * 32;
    const int blocks_in_grid = 8;
    
    int num_threads = threads_per_block * blocks_in_grid;

    // Timer initialization
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    int results_size = num_threads;
    //float * data_out_cpu;
	float sum = 0.;
    // Allocating output data on CPU
	//data_out_cpu = (float *) malloc(results_size*sizeof(float));

	// Allocating output data on GPU
	float *data_gpu;
	CUDA_SAFE_CALL(cudaMalloc(&data_gpu, results_size*sizeof(float)));

    // Start timer
    CUDA_SAFE_CALL(cudaEventRecord(start, 0));

    // Execute kernel
	summation_kernel<threads_per_block><<<blocks_in_grid, threads_per_block>>>(data_size, data_gpu);
	reduce_grid<blocks_in_grid/2><<<1, blocks_in_grid/2>>>(blocks_in_grid, data_size/blocks_in_grid, data_gpu);

    // Stop timer
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    // Get results back
	//CUDA_SAFE_CALL(cudaMemcpy(data_out_cpu, data_gpu, num_threads*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&sum, data_gpu, sizeof(float), cudaMemcpyDeviceToHost));
	
    // Finish reduction
	/* for(int i = 0; i < results_size; i+=threads_per_block)
		sum += data_out_cpu[i]; */
    
    // Cleanup
    //free(data_out_cpu);
    CUDA_SAFE_CALL(cudaFree(data_gpu));
    
    
    printf("GPU results:\n");
    printf(" Sum: %f\n", sum);
    
    float elapsedTime;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));	// In ms

    double total_time = elapsedTime / 1000.;	// s
    double time_per_iter = total_time / (double)data_size;
    double bandwidth = sizeof(float) / time_per_iter; // B/s
    
    printf(" Total time: %g s,\n Per iteration: %g ns\n Throughput: %g GB/s\n",
    	total_time,
    	time_per_iter * 1.e9,
    	bandwidth / 1.e9);
  
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
    return 0;
}

