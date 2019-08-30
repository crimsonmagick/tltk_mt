#include <stdio.h>

__global__
void predicate(long n, float a,float bound,float *traces)
{
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) traces[i] = a*traces[i] - bound;
}


void predicate_setup(float* cpu_traces, float A, float bound,long length){
    float *gpu_traces;
    if(cudaMalloc(&gpu_traces, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
        
    cudaMemcpy(gpu_traces, cpu_traces, length*sizeof(float), cudaMemcpyHostToDevice);
    predicate<<<1, length>>>(length, A,bound,gpu_traces);
    cudaMemcpy(cpu_traces, gpu_traces, length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_traces);
}

