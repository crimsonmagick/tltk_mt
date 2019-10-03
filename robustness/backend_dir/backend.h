#ifndef BACKEND_H
#define BACKEND_H

#include "osqp.h"


void c_not(float* robustness,long length);
void c_or(float* left_robustness, float* right_robustness, long length);
void c_and(float* left_robustness, float* right_robustness, long length);
float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
void c_one_dim_pred(float* traces, float A, float bound,long length);
void higher_dim_pred(c_int n, c_int m, double* q,double* l, double* u, double **init_A, double **init_P, float* traces, long length);

// Auxiliary functions

long search_sorted(float* time_stamps,float time,long start_lower_index,long length);
csc* array_to_csc(c_int m, c_int n, c_float **A);
long find_min(float* array, long start_index, long end_index);
long find_max(float* array, long start_index, long end_index);

//=======
//long find_min(float* array, long length);
//long find_max(float* array, long length);
//minmax find_min_max(float* array, long length);
//>>>>>>> 88b0e505fefb0afa4cdabf5ef68eb257d728420b

#endif

