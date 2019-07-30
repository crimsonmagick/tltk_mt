
#ifndef BACKEND_H
#define BACKEND_H

void c_not(float* robustness,long length);
void c_or(float* left_robustness, float* right_robustness, long length);
void c_and(float* left_robustness, float* right_robustness, long length);
#endif
