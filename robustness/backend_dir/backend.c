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
}

int main(){
    float tst[5] = {1,3,6,7,9};
    long spot = search_sorted(tst,7,0,5);
    printf("%ld\n",spot);

}
