
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int data_chunk = data_size / (blockDim.x * gridDim.x);
	
	float result = 0.0;
	
	if(tid%2)
		for(int i = data_chunk*2*tid+1; i >= data_chunk*2*(tid-1)+1; i-=2)
			result = result-1.0/(i+1);
	else
		for(int i = data_chunk*2*(tid+1); i >= data_chunk*2*tid; i-=2)
			result = result+1.0/(i+1);
	data_out[tid] = result;
}


