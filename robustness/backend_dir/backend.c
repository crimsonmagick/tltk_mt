#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "backend.h"



void  c_not(float* robustness,long length){
    int i;
    for(i = 0; i < length; i++){
        *(robustness + i) = -1 * *(robustness + i);
    }
}

void c_or(float* left_robustness, float* right_robustness, long length){
    int i;
    for(i=0; i < length; i++){
        if(*(left_robustness + i) < *(right_robustness + i)){
            *(left_robustness + i) = *(right_robustness + i);
        }
    }
}

void c_and(float* left_robustness, float* right_robustness, long length){
    int i;
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
 
      	middle = (lower_index + upper_index)/2;
   	}
   	return middle;
}

long find_min(float* array, long length){
	long i, index;
	float min;
	min = *array;
	index = 0;
	for (i = 1; i < length; i++){
		if (*(array + i) < min){
			index = i;
			min = *(array + i);
		}
	}
	return index;
}

long find_max(float* array, long length){
	long i, index;
	float max;
	max = *array;
	index = 0;
	for (i = 1; i < length; i++){
		if (*(array + i) > max){
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

minmax find_min_max(float* array, long length){
	minmax results;
	int i;
	if (*array > *(array+1)){
		results.min = *(array + 1);
		results.max = *array;
	}else{
		results.max = *(array + 1);
		results.min = *array;
	}
	for (i = 0; i < length; i++){
		if (*(array + i) > results.max)
			results.max = *(array + i);
		else if (*(array + i) < results.min)
			results.min = *(array + i);
	}
	return results;
}

int main(){
    float tst[5] = {1,3,6,7,9};
    long spot = search_sorted(tst,5.9,0,5);
    printf("%ld\n",spot);
    /*
	printf("%ld\n",find_min(tst, 5));
	printf("%ld\n",find_max(tst, 5));
	minmax res = find_min_max(tst, 5);
	printf("%f %f\n", res.min, res.max); */
}
