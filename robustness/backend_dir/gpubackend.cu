#include <stdio.h>
#include <time.h>

const int gpu_threads = 1024;
const int gpu_blocks = 1024;

__device__  long search_sorted(float* time_stamps,float time,long start_lower_index,long length){
    long lower_index = start_lower_index;
    long upper_index = length - 1;
    
    long middle = (lower_index + upper_index) / 2; 
 
    while (lower_index <= upper_index) {
        if (*(time_stamps + middle) < time)
            lower_index = middle + 1;    
        else if (*(time_stamps + middle) == time) {
            break;
        }
        else
            upper_index = middle - 1;

        middle = (lower_index + upper_index) /  2;
   	}
   	return middle;
}

__device__  long find_min(float* array, long start_index, long end_index){
    long i, index;
    float min;
    min = *array;
    index = 0;
    for (i = start_index; i <= end_index; i++){
        if (*(array + i) < min){
            index = i;
            min = *(array + i);
        }
    }
    return index;
}

__device__  long find_max(float* array, long start_index, long end_index){
    long i, index;
    float max;
    max = *array;
    index = 0;
    for (i = start_index; i <= end_index; i++){
        if(*(array + i) > max){
            index = i;
            max = *(array + i);
        }
    }
    return index;
}


__global__ void predicate(long length,long sub_length,float a,float bound,float *traces){
    long start_time_step = (blockIdx.x*blockDim.x + threadIdx.x) * sub_length;
    long current_time_step;
    if (start_time_step < length){ 
        for(current_time_step = 0; current_time_step < sub_length && (current_time_step + start_time_step) < length; current_time_step++){
            traces[current_time_step+start_time_step] = a*traces[current_time_step+start_time_step] - bound;
        }
    }
}

__global__ void gpu_and_kernal(long length,long sub_length,float *left_robustness, float *right_robustness){
    long start_time_step = (blockIdx.x*blockDim.x + threadIdx.x) * sub_length;
    long current_time_step;
    if (start_time_step < length){ 
        for(current_time_step = 0; current_time_step < sub_length && (current_time_step + start_time_step) < length; current_time_step++){
            if(*(left_robustness + (current_time_step + start_time_step)) > *(right_robustness + (current_time_step + start_time_step))){
                *(left_robustness + (current_time_step + start_time_step)) = *(right_robustness + (current_time_step + start_time_step));
            }
        }
    }
}

__global__  void gpu_finally_kernal(long length,long sub_length,float lower_time_bound,float upper_time_bound ,float* robustness, float* time_stamps, float* finally_robustness){
    long current_time_step;
    long start_time_step = (blockIdx.x*blockDim.x + threadIdx.x) * sub_length;
    if (start_time_step < length){
        for(current_time_step = 0; current_time_step < sub_length && (current_time_step + start_time_step) < length; current_time_step++){
            float lower_bound = *(time_stamps + (current_time_step + start_time_step)) + lower_time_bound;
            float upper_bound = *(time_stamps + (current_time_step + start_time_step)) + upper_time_bound;
            
            long upper_bound_index = search_sorted(time_stamps,upper_bound,(current_time_step + start_time_step),length);
            
            long lower_bound_index; 
                
            if(lower_time_bound == 0){
                lower_bound_index = (current_time_step + start_time_step);
            }
            else{
                lower_bound_index = search_sorted(time_stamps,lower_bound,(current_time_step + start_time_step),length);
            }
            if(lower_bound_index == upper_bound_index){
                *(finally_robustness + (current_time_step + start_time_step)) = *(robustness + lower_bound_index);
            }
            else{
                long max_index = find_max(robustness,lower_bound_index,upper_bound_index);
                *(finally_robustness + (start_time_step + current_time_step)) = *(robustness + max_index);
            }
        }
    }
}

__global__  void gpu_global_kernal(long length,long sub_length,float lower_time_bound,float upper_time_bound ,float* robustness, float* time_stamps, float* finally_robustness){
    long current_time_step;
    long start_time_step = (blockIdx.x*blockDim.x + threadIdx.x) * sub_length;
    if (start_time_step < length){
        for(current_time_step = 0; current_time_step < sub_length && (current_time_step + start_time_step) < length; current_time_step++){
            float lower_bound = *(time_stamps + (current_time_step + start_time_step)) + lower_time_bound;
            float upper_bound = *(time_stamps + (current_time_step + start_time_step)) + upper_time_bound;
            
            long upper_bound_index = search_sorted(time_stamps,upper_bound,(current_time_step + start_time_step),length);
            
            long lower_bound_index; 
                
            if(lower_time_bound == 0){
                lower_bound_index = (current_time_step + start_time_step);
            }
            else{
                lower_bound_index = search_sorted(time_stamps,lower_bound,(current_time_step + start_time_step),length);
            }
            if(lower_bound_index == upper_bound_index){
                *(finally_robustness + (current_time_step + start_time_step)) = *(robustness + lower_bound_index);
            }
            else{
                long min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                *(finally_robustness + (start_time_step + current_time_step)) = *(robustness + min_index);
            }
        }
    }
}

void predicate_setup(float* cpu_traces, float A, float bound,long length){
    float *gpu_traces;
    if(cudaMalloc(&gpu_traces, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");

    long blocks = (length / gpu_blocks) + 1; //divide by 1024 to the 5.2 compute standard of my 980ti and add 1 encase the division rounded down
    long sub_length = (blocks / gpu_blocks) + 1;
    if(blocks > gpu_blocks){
        blocks = gpu_blocks;
    }

    cudaMemcpy(gpu_traces, cpu_traces, length*sizeof(float), cudaMemcpyHostToDevice);
    predicate<<<blocks, gpu_threads>>>(length, sub_length,A,bound,gpu_traces);
    cudaMemcpy(cpu_traces, gpu_traces, length*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_traces);
}

void c_finally_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length){
    float *gpu_traces;
    float *gpu_time_stamps;
    float *gpu_results;
    
    if(cudaMalloc(&gpu_traces, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
        
    if(cudaMalloc(&gpu_time_stamps, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
        
    if(cudaMalloc(&gpu_results, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");

    cudaMemcpy(gpu_results, results, length*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_traces, cpu_traces, length*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_time_stamps, cpu_time_stamps, length*sizeof(float), cudaMemcpyHostToDevice);
    
    long blocks = (length / gpu_blocks) + 1; //divide by 1024 to the 5.2 compute standard of my 980ti and add 1 encase the division rounded down
    long sub_length = (blocks / gpu_blocks) + 1;
    if(blocks > gpu_blocks){
        blocks = gpu_blocks;
    }

    gpu_finally_kernal<<<blocks,gpu_threads>>>(length,sub_length,lower_time_bound, upper_time_bound,gpu_traces,gpu_time_stamps,gpu_results);


    
    //float *cpu_results; 
    //if((cpu_results = (float*)malloc(length*sizeof(float))) < 0)
    //    perror("CPU MEM ERROR:");

    if(cudaMemcpy(results, gpu_results, length*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        perror("GPU COPY ERROR");
    
    cudaFree(gpu_results);
    cudaFree(gpu_traces);
    cudaFree(gpu_time_stamps);
}


void c_global_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length){
    float *gpu_traces;
    float *gpu_time_stamps;
    float *gpu_results;
    
    if(cudaMalloc(&gpu_traces, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
        
    if(cudaMalloc(&gpu_time_stamps, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");
        
    if(cudaMalloc(&gpu_results, length*sizeof(float)) != cudaSuccess)
        perror("GPU MEM ERROR");

    cudaMemcpy(gpu_results, results, length*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_traces, cpu_traces, length*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_time_stamps, cpu_time_stamps, length*sizeof(float), cudaMemcpyHostToDevice);
    
    long blocks = (length / gpu_blocks) + 1; //divide by 1024 to the 5.2 compute standard of my 980ti and add 1 encase the division rounded down
    long sub_length = (blocks / gpu_blocks) + 1;
    if(blocks > gpu_blocks){
        blocks = gpu_blocks;
    }

    gpu_global_kernal<<<blocks,gpu_threads>>>(length,sub_length,lower_time_bound, upper_time_bound,gpu_traces,gpu_time_stamps,gpu_results);


    
    //float *cpu_results; 
    //if((cpu_results = (float*)malloc(length*sizeof(float))) < 0)
    //    perror("CPU MEM ERROR:");

    if(cudaMemcpy(results, gpu_results, length*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        perror("GPU COPY ERROR");
    
    cudaFree(gpu_results);
    cudaFree(gpu_traces);
    cudaFree(gpu_time_stamps);
}

int main(){
    long length = 10000000;
    float *traces = (float*)malloc(length*sizeof(float)); 
    float *time_stamps = (float*)malloc(length*sizeof(float));
    float *results = (float*)malloc(length*sizeof(float));
    long i;
    double time_spent = 0;
    for(i = 0; i < length;i++){
        traces[i] = 2;
        time_stamps[i] = i;
    }
    traces[30] = -1;
    clock_t begin = clock();
    //predicate_setup(traces, 2.0f, 0.0f,length);
    c_global_gpu(traces,time_stamps,results,0,100,length);
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%g\n",time_spent);
    //results[0] = 3.0f;
    printf("%f\n", results[0]);
}
