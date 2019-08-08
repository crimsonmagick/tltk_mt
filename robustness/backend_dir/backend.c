#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "backend.h"



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

// In case we can do a min-max simultaneously,
// the following way is by doing pairwise comparisons using a struct.
// It is much faster but too situational. I am adding it just in case.
// Feel free to erase it (Rania 7/31)

//minmax find_min_max(float* array, long length){
    //minmax results;
    //int i;
    //if (*array > *(array+1)){
        //results.min = *(array + 1);
        //results.max = *array;
    //}else{
        //results.max = *(array + 1);
        //results.min = *array;
    //}
    //for (i = 0; i < length; i++){
        //if (*(array + i) > results.max)
            //results.max = *(array + i);
        //else if (*(array + i) < results.min)
            //results.min = *(array + i);
    //}
    //return results;
//}

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
    float tst[5] = {-1,-1,6,2,4};
    float tst2[5] = {-1,-1,3,3,2};
    float time[5] = {1,2,3,4,5};
    /*long spot = search_sorted(tst,5.9,0,5);
    printf("%ld\n",spot);*/
    float upper_bound = 1;
    float* robustness;
    robustness =  c_until(0,upper_bound,tst,tst2,time,5);
    int i;
    for(i = 0; i < 5; i++){
        printf("%f ",robustness[i]);
    }
    printf("\n");
    //printf("%ld\n",find_min(tst, 5));
    //printf("%ld\n",find_max(tst, 5));
    //minmax res = find_min_max(tst, 5);
    //printf("%f %f\n", res.min, res.max); 
    return 0;
}
