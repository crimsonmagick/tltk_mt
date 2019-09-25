#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

void predicate_setup(float* cpu_traces, float A, float bound,long length);
void c_not_gpu(float* robustness, long length);
void c_or_gpu(float* left_robustness, float* right_robustness, long length);
void predicate_setup(float* cpu_traces, float A, float bound,long length);
void c_and_gpu(float* left_robustness, float* right_robustness, long length);
void c_finally_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length);
void c_global_gpu(float* cpu_traces, float* cpu_time_stamps, float* results,float lower_time_bound, float upper_time_bound,long length);
float* c_until_gpu(float lower_time_bound, float upper_time_bound, float* left_robustness, float* right_robustness, float* time_stamps,float* results, long length);

#endif

