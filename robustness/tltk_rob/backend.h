#ifndef BACKEND_H
#define BACKEND_H

#include "omp.h"

void c_not(float* robustness,long length);
void c_or(float* left_robustness, float* right_robustness, long length);
void c_and(float* left_robustness, float* right_robustness, long length);
void c_not_threaded(float* robustness,long length);
void c_or_threaded(float* left_robustness, float* right_robustness, long length);
void c_and_threaded(float* left_robustness, float* right_robustness, long length);
void c_next_no_malloc(float* robustness,long length);

float* c_finally(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_finally_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness ,long length);
float* c_global(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_global_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness ,long length);
float* c_until(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
float* c_until_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness,float* until_robustness, float* time_stamps, long length);
float* c_finally_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_finally_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness ,long length);
void finally_thread_task(long start_index,long end_index,float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* finally_robustness,long length);
float* c_global_threaded(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps, long length);
float* c_global_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length);
void global_thread_task(long start_index,long end_index,float lower_time_bound, float upper_time_bound, float* robustness, float* time_stamps,float* global_robustness,long length);
float* c_until_threaded(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps, long length);
float* c_until_threaded_no_malloc(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* until_robustness ,long length);
void c_one_dim_pred(float* traces, float A, float bound,long length);
void c_one_dim_pred_threaded(float* traces, float A, float bound,long length);

double* higher_dim_pred(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length, double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr, double* init_A);
double* higher_dim_pred_threaded(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length, double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr, double* init_A);

// Auxiliary functions

long search_sorted(float* time_stamps,float time,long start_lower_index,long length);
long find_min(float* array, long start_index, long end_index);
long find_max(float* array, long start_index, long end_index);

//=======
//long find_min(float* array, long length);
//long find_max(float* array, long length);
//minmax find_min_max(float* array, long length);
//>>>>>>> 88b0e505fefb0afa4cdabf5ef68eb257d728420b

#endif

