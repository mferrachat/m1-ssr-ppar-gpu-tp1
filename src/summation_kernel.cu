
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	unsigned int tid = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tid;
	int data_chunk = data_size / (blockDim.x * gridDim.x);
	
	float result = 0.0;
	
	for(int i = data_chunk*(id+1); i >= data_chunk*id; i--)
		result = i&1 ? result-1.0/(i+1) : result+1.0/(i+1);
	data_out[id] = result;
	
	for(int j = 2; j < blockDim.x; j *= 2)
		if(tid < (blockDim.x/j))
			data_out[id] += data_out[id+blockDim.x/j];
}

__global__ void reduce_grid(int data_size, int data_chunk, float * data_out)
{
	unsigned int tid = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tid;
	
	for(int j = 1; j < blockDim.x; j *= 2)
		if(tid < (blockDim.x/j))
			data_out[id*data_chunk] += data_out[(id+blockDim.x/j)*data_chunk];
}