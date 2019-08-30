#include <stdio.h>

__global__
void predicate(long n, float a, float *traces, float *bounds)
{
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) bounds[i] = a*traces[i] - bounds[i];
}

int main(){
    long N = 100000000;
    float *cpu_traces, *y, *gpu_traces, *gpu_bounds;
    if((cpu_traces = (float*)malloc(N*sizeof(float))) < 0)
        perror("CPU MEM ERROR");
    if((y = (float*)malloc(N*sizeof(float))) < 0)
        perror("CPU MEM ERROR");

    if(cudaMalloc(&gpu_traces, N*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
    
    if(cudaMalloc(&gpu_bounds, N*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");

    for (long i = 0; i < N; i++) {
        cpu_traces[i] = 2.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(gpu_traces, cpu_traces, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bounds, y, N*sizeof(float), cudaMemcpyHostToDevice);

   
    predicate<<<1, N>>>(N, 2.0f, gpu_traces, gpu_bounds);
    cudaMemcpy(y, gpu_bounds, N*sizeof(float), cudaMemcpyDeviceToHost);

  //float maxError = 0.0f;
    //for (int i = 0; i < N; i++){
        //if(y[i] != 2.0f)
            //maxError = y[i];
        //printf("%f\n",y[i]);
    //}
  //printf("Max error: %f\n", maxError);

    cudaFree(gpu_traces);
    cudaFree(gpu_bounds);
    free(cpu_traces);
    free(y);
}
