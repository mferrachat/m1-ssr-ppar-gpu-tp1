
// GPU kernel
__global__ void summation_kernel(int data_size, float * data_out)
{
	float result = 0.0;
	for(int i = ; i >= ; i--)
		*data_out = i%2 ? result-1.0/(i+1) : result+1.0/(i+1);
}


