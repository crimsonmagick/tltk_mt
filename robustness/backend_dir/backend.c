#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "backend.h"
#include <omp.h>

float glower_time_bound;
float gupper_time_bound;
float* grobustness;
float* gtime_stamps;
long glength;
float *gglobal_robustness;
float *gfinally_robustness;

void  c_not(float* robustness,long length){
    long i;
    for(i = 0; i < length; i++){
        *(robustness + i) = -1 * *(robustness + i);
    }
}

void c_or(float* left_robustness, float* right_robustness, long length){
    long i;
    for(i=0; i < length; i++){
        if(*(left_robustness + i) < *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}

void c_and(float* left_robustness, float* right_robustness, long length){
    long i;
    for(i=0; i < length; i++){
        if(*(left_robustness + i) > *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}

/*
long search_sorted(float* time_stamps,float time,long start_lower_index,long length){
    long lower_index = start_lower_index;
    long upper_index = length;
    long mid_index;
    int flag = 0;
    while(flag == 0){
        mid_index = (lower_index + upper_index) / 2; //Rounds down to lower index if odd
        if(time < *(time_stamps + mid_index)) {
            upper_index = mid_index;
        }
        else if(*(time_stamps + mid_index) < time){
            lower_index = mid_index;
        }
        else{
            flag = 1;
        }
    }
    return mid_index;
}*/

// search sorted attempt (Rania 8/1)

long search_sorted(float* time_stamps,float time,long start_lower_index,long length){
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
float max(float left, float right){
    if(left > right){
        return left;
    }
    return right;
}

float min(float left, float right){
    if(left < right){
        return left;
    }
    return right;
}


long find_min(float* array, long start_index, long end_index){
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

long find_max(float* array, long start_index, long end_index){
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

void finally_task(long current_time_step){
	float lower_bound = *(gtime_stamps + current_time_step) + glower_time_bound;
	float upper_bound = *(gtime_stamps + current_time_step) + gupper_time_bound;
        long upper_bound_index = search_sorted(gtime_stamps,upper_bound,current_time_step,glength);
	long lower_bound_index; 
	if(glower_time_bound == 0){
		lower_bound_index = current_time_step;
	}
	else{
		lower_bound_index = search_sorted(gtime_stamps,lower_bound,current_time_step,glength);
	}

	if(lower_bound_index == upper_bound_index){
		*(gfinally_robustness + current_time_step) = *(grobustness + lower_bound_index);
	}
	else{
		long max_index = find_max(grobustness,lower_bound_index,upper_bound_index);
		*(gfinally_robustness + current_time_step) = *(grobustness + max_index);
	}
}


float* c_finally_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
	long i;
	float max;
	float* finally_robustness = (float*) malloc(length * sizeof(float));
	gfinally_robustness = finally_robustness;
	if(finally_robustness == NULL){
		perror("Error: finally could not malloc memory");
		exit(-1);
	}

	if(lower_time_bound == 0 && isinf(upper_time_bound)){
		max = *(robustness + (length - 1));
		for(i = length - 1; i >= 0; i--){
		    if(*(robustness + i) > max){
			max = *(robustness + i);
		    }
		    *(gfinally_robustness + i) = max;
		}
	}
	else{
		long current_time_step;
		#pragma omp
		#pragma omp single
		#pragma omp taskloop num_tasks(100)
		for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
		    float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
		    float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
		    //long search_sorted(float* time_stamps,float time,long start_lower_index,long length)
		    long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
		    long lower_bound_index; 
		    
		    if(lower_time_bound == 0){
		        lower_bound_index = current_time_step;
		    }
		    else{
		        lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
		    }
		    
		    if(lower_bound_index == upper_bound_index){
		        *(finally_robustness + current_time_step) = *(robustness + lower_bound_index);
		    }
		    else{
		        long max_index = find_max(robustness,lower_bound_index,upper_bound_index);
		        *(finally_robustness + current_time_step) = *(robustness + max_index);
		    }
		
		}
	}
	
	return gfinally_robustness;
}
float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
    long i;
    float max;
    float* finally_robustness = (float*) malloc(length * sizeof(float));
    
    if(finally_robustness == NULL){
        perror("Error: finally could not malloc memory");
        exit(-1);
    }
    
    if(lower_time_bound == 0 && isinf(upper_time_bound)){
        max = *(robustness + (length - 1));
        for(i = length - 1; i >= 0; i--){
            if(*(robustness + i) > max){
                max = *(robustness + i);
            }
            *(finally_robustness + i) = max;

        }
    }
    else{
        long current_time_step;
        for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
            //long search_sorted(float* time_stamps,float time,long start_lower_index,long length)
            long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
            long lower_bound_index; 
            
            if(lower_time_bound == 0){
                lower_bound_index = current_time_step;
            }
            else{
                lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
            }
            
            if(lower_bound_index == upper_bound_index){
                *(finally_robustness + current_time_step) = *(robustness + lower_bound_index);
            }
            else{
                long max_index = find_max(robustness,lower_bound_index,upper_bound_index);
                *(finally_robustness + current_time_step) = *(robustness + max_index);
            }
        
        }
    }
    return finally_robustness;
}


float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
    long i;
    float min;
    float* global_robustness = (float*) malloc(length * sizeof(float));
    
    if(global_robustness == NULL){
        perror("Error: global could not malloc memory");
        exit(-1);
    }
    
    if(lower_time_bound == 0 && isinf(upper_time_bound)){
        min = *(robustness + (length - 1));
        for(i = length - 1; i >= 0; i--){
            if(*(robustness + i) < min){
                min = *(robustness + i);
            }
            *(global_robustness + i) = min;

        }
    }
    else{
        long current_time_step;
        for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
            //long search_sorted(float* time_stamps,float time,long start_lower_index,long length)
            long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
            long lower_bound_index; 
            
            if(lower_time_bound == 0){
                lower_bound_index = current_time_step;
            }
            else{
                lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
            }
            
            if(lower_bound_index == upper_bound_index){
                *(global_robustness + current_time_step) = *(robustness + lower_bound_index);
            }
            else{
                long min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                *(global_robustness + current_time_step) = *(robustness + min_index);
            }
        
        }
    }
    return global_robustness;
}

void global_task(long current_time_step){
	
	float lower_bound = *(gtime_stamps + current_time_step) + glower_time_bound;
	float upper_bound = *(gtime_stamps + current_time_step) + gupper_time_bound;
	//long search_sorted(float* time_stamps,float time,long start_lower_index,long length)
	long upper_bound_index = search_sorted(gtime_stamps,upper_bound,current_time_step,glength);
	long lower_bound_index; 

	if(glower_time_bound == 0){
		lower_bound_index = current_time_step;
	}
	else{
		lower_bound_index = search_sorted(gtime_stamps,lower_bound,current_time_step,glength);
	}

	if(lower_bound_index == upper_bound_index){
		*(gglobal_robustness + current_time_step) = *(grobustness + lower_bound_index);
	}
	else{
		long min_index = find_min(grobustness,lower_bound_index,upper_bound_index);
		*(gglobal_robustness + current_time_step) = *(grobustness + min_index);
	}
}

float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
    long i;
    float min;
    float* global_robustness = (float*) malloc(length * sizeof(float));
    gglobal_robustness = global_robustness;
    if(global_robustness == NULL){
        perror("Error: global could not malloc memory");
        exit(-1);
    }
    
    if(lower_time_bound == 0 && isinf(upper_time_bound)){
        min = *(robustness + (length - 1));
        for(i = length - 1; i >= 0; i--){
            if(*(robustness + i) < min){
                min = *(robustness + i);
            }
            *(gglobal_robustness + i) = min;

        }
    }
    else{
        long* current_time_step = (long*)malloc((length) * sizeof(long));
	glower_time_bound = lower_time_bound;
	gupper_time_bound = upper_time_bound;
	grobustness = robustness;
	gtime_stamps = time_stamps;
	glength = length;
        #pragma omp parallel
	#pragma omp single
	#pragma omp taskloop num_tasks(1000)
	for(long counter = length - 1; counter >= 0; counter--){
		
		current_time_step[counter] = counter;
		global_task(current_time_step[counter]);
	
	}
    }
    return global_robustness;
}

void c_one_dim_pred(float* traces, float A, float bound,long length){
    long i;
    for(i = 0; i < length; i++){
        *(traces + i)  =  *(traces + i) * A - bound;
    }
}

float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length){
    float* until_robustness = (float*) malloc(length * sizeof(float));
    if(lower_time_bound == 0 && isinf(upper_time_bound)){
        float last_robustness = -INFINITY;
        long current_time_step;
        for(current_time_step = length - 1; current_time_step >= 0; current_time_step--){
            last_robustness = max(min(last_robustness,left_robustness[current_time_step]),right_robustness[current_time_step]);
            until_robustness[current_time_step] = last_robustness;
        }
    }
    else{
        long current_time_step;
        float last_robustness = -INFINITY;
        for(current_time_step = length-1; current_time_step >= 0; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
            long lower_bound_index;
            if(lower_time_bound == 0){
                lower_bound_index = current_time_step;
            }
            else{
                lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
            }
            long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
            
            float min_robustness;
            
            if(lower_bound_index == current_time_step){
                min_robustness = *(left_robustness+lower_bound_index);
            }
            else{
                long min_robustness_index;
                min_robustness_index = find_min(left_robustness,current_time_step,lower_bound_index);
                min_robustness = *(left_robustness + min_robustness_index);
            }
            long bounded_index;
            
            for(bounded_index = lower_bound_index; bounded_index <= upper_bound_index; bounded_index++){
                    last_robustness = max(last_robustness,min(right_robustness[bounded_index],min_robustness));
                    min_robustness = min(min_robustness,left_robustness[bounded_index]);
            }
            *(until_robustness + current_time_step) = last_robustness;
            last_robustness = -INFINITY;
        }
    }
    return until_robustness;
}  



int main(){
    long length = 100000000;
    float *left_traces = (float*)malloc(length*sizeof(float));
    float *right_traces = (float*)malloc(length*sizeof(float));
    float *time_stamps = (float*)malloc(length*sizeof(float));
    float *results = (float*)malloc(length*sizeof(float));
    long i;
    double time_spent = 0;
    for(i = 0; i < length;i++){
        left_traces[i] = 1;
        right_traces[i] = 2;
        time_stamps[i] = i;
    }
    printf("boobies\n");
    clock_t begin = clock();
    //predicate_setup(traces, 2.0f, 0.0f,length);
    results = c_finally_threaded(0.0f,100.0f, left_traces,time_stamps, length);
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%g\n",time_spent);
    //results[0] = 3.0f;


    printf("%f\n", results[1]);
}

