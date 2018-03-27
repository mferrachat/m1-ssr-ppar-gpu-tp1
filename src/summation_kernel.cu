
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	int tid = blockIdx.x * blockDim.x +threadIdx.x;
	int data_chunk = data_size / (blockDim.x * gridDim.x);
	
	float result = 0.0;
	
	for(int i = data_chunk*(tid+1); i >= data_chunk*tid; i--)
		result = i%2 ? result-1.0/(i+1) : result+1.0/(i+1);
	data_out[tid] = result;
}


