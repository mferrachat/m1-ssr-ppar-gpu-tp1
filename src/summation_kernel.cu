
// GPU kernel
template <unsigned int blockSize> __global__ void summation_kernel(int data_size, float * data_out)
{
	unsigned int tid = threadIdx.x;
	int id = blockIdx.x * blockSize + tid;
	int data_chunk = data_size / (blockDim.x * gridDim.x);
	
	float result = 0.0;
	
	for(int i = data_chunk*(id+1); i >= data_chunk*id; i--)
		result = i&1 ? result-1.0/(i+1) : result+1.0/(i+1);
	data_out[id] = result;
	
	if(blockSize >= 1024)
	{
		if (tid < 512)
			data_out[id] += data_out[id+512];
		__syncthreads();
	}
	
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

template <unsigned int blockSize> __global__ void reduce_grid(int data_size, int data_chunk, float * data_out)
{
	unsigned int tid = threadIdx.x;
	int id = blockIdx.x * blockSize * 2 + tid;
	int i = id;
	int gridSize = blockSize * 2 * gridDim.x;
	
	while(i < data_size)
	{
		data_out[id*data_chunk] += data_out[(id+blockSize)*data_chunk];
		i += gridSize;
	}
	
	if(blockSize >= 1024)
	{
		if ((tid < 512) && (data_size >= 1024))
			data_out[id*data_chunk] += data_out[(id+512)*data_chunk];
		__syncthreads();
	}
	
	if(blockSize >= 512)
	{
		if ((tid < 256) && (data_size >= 512))
			data_out[id*data_chunk] += data_out[(id+256)*data_chunk];
		__syncthreads();
	}
	
	if(blockSize >= 256)
	{
		if ((tid < 128) && (data_size >= 256))
			data_out[id*data_chunk] += data_out[(id+128)*data_chunk];
		__syncthreads();
	}
	
	if(blockSize >= 128)
	{
		if ((tid < 64) && (data_size >= 128))
			data_out[id*data_chunk] += data_out[(id+64)*data_chunk];
		__syncthreads();
	}
	
	if(tid < 32)
	{
		/* Le premier if est évalué à la compilation.
		   Le deuxième à l'exécution (ils sont pas séparé pour rien!!!).*/
		if(blockSize >= 64) if(data_size >= 64) data_out[id*data_chunk] += data_out[(id+32)*data_chunk];
		if(blockSize >= 32) if(data_size >= 32) data_out[id*data_chunk] += data_out[(id+16)*data_chunk];
		if(blockSize >= 16) if(data_size >= 16) data_out[id*data_chunk] += data_out[(id+ 8)*data_chunk];
		if(blockSize >=  8) if(data_size >=  8) data_out[id*data_chunk] += data_out[(id+ 4)*data_chunk];
		if(blockSize >=  4) if(data_size >=  4) data_out[id*data_chunk] += data_out[(id+ 2)*data_chunk];
		if(blockSize >=  2) if(data_size >=  2) data_out[id*data_chunk] += data_out[(id+ 1)*data_chunk];
	}
}