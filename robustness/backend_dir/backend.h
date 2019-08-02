#ifndef BACKEND_H
#define BACKEND_H

typedef struct minmax{
	float min;
	float max;
} minmax;

void c_not(float* robustness,long length);
void c_or(float* left_robustness, float* right_robustness, long length);
void c_and(float* left_robustness, float* right_robustness, long length);

// Auxiliary functions

long search_sorted(float* time_stamps,float time,long start_lower_index,long length);
long find_min(float* array, long length);
long find_max(float* array, long length);
minmax find_min_max(float* array, long length);

#endif

