//C code for processing MTL robustness fromulas
//Uses the dp_taliro algorithms to process robustnesses

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "backend.h"
#include <omp.h>
#include <stdbool.h>

#include <unistd.h>

const int True = 0;
const int False = 1;

//  Processes mtl not, and stores results in *robustness
//      robustness: a float pointer of robustnesses
//      length: how many robustness timesteps are stored in robustness
void  c_not(float* robustness,long length){
    long i;
    for(i = 0; i < length; i++){
        *(robustness + i) = -1 * *(robustness + i);
    }
}

void  c_not_threaded(float* robustness,long length){
    long i;
    #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
    for(i = 0; i < length; i++){
        *(robustness + i) = -1 * *(robustness + i);
    }
}


//  Processes mtl or, and stores results in left_robustness
//      left_robustness: a float pointer of robustnesses
//      right_robustness: a float pointer of robustnesses
//      length: how many robustness timesteps are stored in left_robustness and right_robustness
void c_or(float* left_robustness, float* right_robustness, long length){
    long i;
    for(i=0; i < length; i++){
        if(*(left_robustness + i) < *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}

void c_or_threaded(float* left_robustness, float* right_robustness, long length){
    long i;
    #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
    for(i=0; i < length; i++){
        if(*(left_robustness + i) < *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}


//  Processes mtl and, and stores results in left_robustness
//      left_robustness: a float pointer of robustnesses
//      right_robustness: a float pointer of robustnesses
//      length: how many robustness timesteps are stored in left_robustness and right_robustness
void c_and(float* left_robustness, float* right_robustness, long length){
    long i;
    for(i=0; i < length; i++){
        if(*(left_robustness + i) > *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}

void c_and_threaded(float* left_robustness, float* right_robustness, long length){
    long i;
    #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
    for(i=0; i < length; i++){
        if(*(left_robustness + i) > *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}


//  Searches time stamps for desired time step using binary search
//      time_stamps: a pointer to the time stamps to be searched
//      time: the target time we are looking for a value near
//      start_lower_index: the lower bound of where we need to search
//      length: how many time steps there are
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

//  Returns the max of two inputs
float max(float left, float right){
    if(left > right){
        return left;
    }
    return right;
}
//  Returns min of two inputs
float min(float left, float right){
    if(left < right){
        return left;
    }
    return right;
}

//returns the index of the minimium value in array
//  array: a float pointer
//  start_index: lower bound on where to search in array
//  end_index: upper bound on where to search in array 
long find_min(float* array, long start_index, long end_index){
    long i, index;
    float min;
    min = *(array+start_index);
    index = 0;
    for (i = start_index; i <= end_index; i++){
        if (*(array + i) < min){
            index = i;
            min = *(array + i);
        }
    }
    return index;
}

//returns the index of the max value in array
//  array: a float pointer
//  start_index: lower bound on where to search in array
//  end_index: upper bound on where to search in array 
long find_max(float* array, long start_index, long end_index){
    long i, index;
    float max;
    max = *(array+start_index);
    index = 0;
    for (i = start_index; i <= end_index; i++){
        if(*(array + i) > max){
            index = i;
            max = *(array + i);
        }
    }
    return index;
}

//  Processes mtl finally useing parrell processing
//      lower_time_bound: a float representing the finally lower time bound
//      upper_time_bound: a float representing the finally upper time bound
//      robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns finally_robustness: an array of length, length storing the results of the finally operation
float* c_finally_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
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
        #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
        for(current_time_step = length - 1; current_time_step >= 0; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
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


float* c_finally_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness ,long length){
    long i;
    float max;
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
        long thread_count = sysconf(_SC_NPROCESSORS_ONLN);
        //long current_time_step;
        long division_length = length / thread_count;
        long overflow_work = length % thread_count;
        long task;
        //printf("task over_flow: %ld\n", overflow_work);
        #pragma omp parallel for num_threads(thread_count)
        for(task = 0; task < thread_count; task++){
            long start_index = task * division_length;
            long end_index = (task + 1) * division_length;
            
            if(task == (thread_count - 1)){
                end_index += overflow_work;
            }
            
            finally_thread_task(start_index,end_index,lower_time_bound,upper_time_bound,robustness,time_stamps,finally_robustness,length);
        }
        
    }
    
    return finally_robustness;
}



//  Processes mtl finally
//      lower_time_bound: a float representing the finally lower time bound
//      upper_time_bound: a float representing the finally upper time bound
//      robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns finally_robustness: an array of length, length storing the results of the finally operation
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


float* c_finally_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness ,long length){
    long i;
    float max;
    
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
        finally_thread_task(0,length,lower_time_bound,upper_time_bound,robustness,time_stamps,finally_robustness,length);
        
        //long current_time_step;
        //for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
            //float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            //float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
            //long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
            //long lower_bound_index; 
            
            //if(lower_time_bound == 0){
                //lower_bound_index = current_time_step;
            //}
            //else{
                //lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
            //}
            
            //if(lower_bound_index == upper_bound_index){
                //*(finally_robustness + current_time_step) = *(robustness + lower_bound_index);
            //}
            //else{
                //long max_index = find_max(robustness,lower_bound_index,upper_bound_index);
                //*(finally_robustness + current_time_step) = *(robustness + max_index);
            //}
        
        //}
    }
    return finally_robustness;
}




//  Processes mtl global
//      lower_time_bound: a float representing the global lower time bound
//      upper_time_bound: a float representing the global upper time bound
//      robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns global_robustness: a pointer to the array of length, length storing the results of the global operation
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

float* c_global_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness ,long length){
    long i;
    float min;
    //float* global_robustness = (float*) malloc(length * sizeof(float));
    
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
        global_thread_task(0,length,lower_time_bound,upper_time_bound,robustness,time_stamps,global_robustness,length);
        //long current_time_step;
        //long previous_lower_bound_index;
        //long min_index = -1;
        //float min = -INFINITY;
        //for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
            //float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            //float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
            //long upper_bound_index = search_sorted(time_stamps,upper_bound,current_time_step,length);
            //long lower_bound_index; 
            
            //if(lower_time_bound == 0){
                //lower_bound_index = current_time_step;
            //}
            //else{
                //lower_bound_index = search_sorted(time_stamps,lower_bound,current_time_step,length);
            //}
            
            //if(lower_bound_index == upper_bound_index){
                //*(global_robustness + current_time_step) = *(robustness + lower_bound_index);
            //}
            //else{
                //if(min_index == -1){
                    //min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                    //*(global_robustness + current_time_step) = *(robustness + min_index);
                //}
                //else if(min_index > upper_bound_index){
                    //min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                    //*(global_robustness + current_time_step) = *(robustness + min_index);
                //}
                //else{
                    //long possible_min_index = find_min(robustness,lower_bound_index,previous_lower_bound_index);
                    //if(*(robustness+possible_min_index) <= min){
                        //min_index = possible_min_index;
                    //}
                    //*(global_robustness + current_time_step) = *(robustness + min_index);
                //}
            //}
            //previous_lower_bound_index = lower_bound_index;
            //min = *(global_robustness + min_index);
        //}
    
    }
    return global_robustness;
}


//  Processes mtl global using parrel processing
//      lower_time_bound: a float representing the global lower time bound
//      upper_time_bound: a float representing the global upper time bound
//      robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns global_robustness: a pointer to the array of length, length storing the results of the global operation
float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length){
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
        #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
        for(current_time_step= length - 1; current_time_step >= 0; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
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

void finally_thread_task(long start_index,long end_index,float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness,long length){
    long current_time_step;
     //printf("Start index: %ld | End index: %ld\n",start_index,end_index - 1);
    long previous_lower_bound_index;
    long max_index = -1;
    float max = -INFINITY;
    for(current_time_step= end_index - 1; current_time_step >= start_index; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
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
                if(max_index == -1){
                    max_index = find_max(robustness,lower_bound_index,upper_bound_index);
                    *(finally_robustness + current_time_step) = *(robustness + max_index);
                }
                else if(max_index > upper_bound_index){
                    max_index = find_max(robustness,lower_bound_index,upper_bound_index);
                    *(finally_robustness + current_time_step) = *(robustness + max_index);
                }
                else{
                    long possible_max_index = find_max(robustness,lower_bound_index,previous_lower_bound_index);
                    if(*(robustness+possible_max_index) <= max){
                        max_index = possible_max_index;
                    }
                    *(finally_robustness + current_time_step) = *(robustness + max_index);
                }
            }
            previous_lower_bound_index = lower_bound_index;
            max = *(robustness + max_index);
    }
}


void global_thread_task(long start_index,long end_index,float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length){
    long current_time_step;
     //printf("Start index: %ld | End index: %ld\n",start_index,end_index - 1);
    long previous_lower_bound_index;
    long min_index = -1;
    float min = INFINITY;
    for(current_time_step= end_index - 1; current_time_step >= start_index; current_time_step--){
            float lower_bound = *(time_stamps + current_time_step) + lower_time_bound;
            float upper_bound = *(time_stamps + current_time_step) + upper_time_bound;
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
                if(min_index == -1){
                    min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                    *(global_robustness + current_time_step) = *(robustness + min_index);
                }
                else if(min_index > upper_bound_index){
                    min_index = find_min(robustness,lower_bound_index,upper_bound_index);
                    *(global_robustness + current_time_step) = *(robustness + min_index);
                }
                else{
                    long possible_min_index = find_min(robustness,lower_bound_index,previous_lower_bound_index);
                    if(*(robustness+possible_min_index) <= min){
                        min_index = possible_min_index;
                    }
                    *(global_robustness + current_time_step) = *(robustness + min_index);
                }
            }
            previous_lower_bound_index = lower_bound_index;
            min = *(robustness + min_index);
    }
}

float* c_global_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length){
    long i;
    float min;
    //float* global_robustness = (float*) malloc(length * sizeof(float));
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
        long thread_count = sysconf(_SC_NPROCESSORS_ONLN);
        //long current_time_step;
        long division_length = length / thread_count;
        long overflow_work = length % thread_count;
        long task;
        //printf("task over_flow: %ld\n", overflow_work);
        #pragma omp parallel for num_threads(thread_count)
        for(task = 0; task < thread_count; task++){
            long start_index = task * division_length;
            long end_index = (task + 1) * division_length;
            
            if(task == (thread_count - 1)){
                end_index += overflow_work;
            }
            
            global_thread_task(start_index,end_index,lower_time_bound,upper_time_bound,robustness,time_stamps,global_robustness,length);
        }
        
    }
    return global_robustness;
}


//  Process 1 dimisional polyhedron A*trace[i] <= bound
//      traces: an array of length, length containing data from simulation
//      A: a float
//      bound: the polyhedron boundary
//      length: how many time steps there are
void c_one_dim_pred(float* traces, float A, float bound,long length){
    long i;
    for(i = 0; i < length; i++){
        *(traces + i)  =  -1*(*(traces + i) * A - bound);
    }
}


void c_one_dim_pred_threaded(float* traces, float A, float bound,long length){
    long i;
    #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
    for(i = 0; i < length; i++){
        *(traces + i)  =  -1*(*(traces + i) * A - bound);
    }
}

//  Processes mtl until (left_robustness Until Right_robustness)
//      lower_time_bound: a float representing the until lower time bound
//      upper_time_bound: a float representing the until upper time bound
//      left_robustness: a float pointer of robustnesses
//      right_robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns until_robustness: a pointer to the array of length, length storing the results of the until operation
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

//  Processes mtl until using parallel processing(left_robustness Until Right_robustness)
//      lower_time_bound: a float representing the until lower time bound
//      upper_time_bound: a float representing the until upper time bound
//      left_robustness: a float pointer of robustnesses
//      right_robustness: a float pointer of robustnesses
//      time_stamps: stores the time stamp for each time step
//      length: how many robustness timesteps are stored in robustness
//      returns until_robustness: a pointer to the array of length, length storing the results of the until operation
float* c_until_threaded(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length){
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
        #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
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

float* c_until_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* until_robustness ,long length){
    //float* until_robustness = (float*) malloc(length * sizeof(float));
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
        #pragma omp parallel for num_threads(sysconf(_SC_NPROCESSORS_ONLN))
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

float* c_until_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness,float* until_robustness, float* time_stamps, long length){
    //float* until_robustness = (float*) malloc(length * sizeof(float));
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

void c_next_no_malloc(float* robustness,long length){
    long current_time_step;
    //this can be changed to pointer manipulation later so there wont have to be a loop
    for(current_time_step = 0; current_time_step < length; current_time_step++){
        if((current_time_step + 1) != length){
            *(robustness + current_time_step) = *(robustness + (current_time_step + 1)); 
        }
    }

}
