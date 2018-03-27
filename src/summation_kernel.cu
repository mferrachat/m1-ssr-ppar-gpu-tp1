
// GPU kernel
template <unsigned int blockSize> __global__ void summation_kernel(int data_size, float * data_out)
{
	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + tid;
	unsigned int data_chunk = data_size / (blockDim.x * gridDim.x);
	
	float result = 0.0;
	
	for(unsigned int i = data_chunk*(id+1); i >= data_chunk*id; i--)
		result = i%2 ? result-1.0/(i+1) : result+1.0/(i+1);
	data_out[id] = result;
	
	if(blockSize >= 512)
	{
		if (tid < 256)
			data_out[id] += data_out[id+256];
		__syncthreads();
	}
	
	if(blockSize >= 256)
	{
		if (tid < 128)
			data_out[id] += data_out[id+128];
		__syncthreads();
	}
	
	if(blockSize >= 128)
	{
		if (tid < 64)
			data_out[id] += data_out[id+64];
		__syncthreads();
	}
	
	if(tid < 32)
	{
		if(blockSize >= 64) data_out[id] += data_out[id+32];
		if(blockSize >= 32) data_out[id] += data_out[id+16];
		if(blockSize >= 16) data_out[id] += data_out[id+ 8];
		if(blockSize >=  8) data_out[id] += data_out[id+ 4];
		if(blockSize >=  4) data_out[id] += data_out[id+ 2];
		if(blockSize >=  2) data_out[id] += data_out[id+ 1];
	}
}

__global__ void reduce_grid()
{
	
}