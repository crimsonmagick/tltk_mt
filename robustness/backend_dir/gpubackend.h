#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

void predicate_setup(float* cpu_traces, float A, float bound,long length);
void c_or_gpu(float* left_robustness, float* right_robustness, long length);

#endif

