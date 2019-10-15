#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "backend.h"
#include <omp.h>
#include <stdbool.h>
#include "osqp.h"


const int True = 0;
const int False = 1;

int in_set(double* left,double* right, long length){
    long i;
    for(i = 0; i <= length; i++){
        if(left[i] > right[i]){
            return False;
        }
    }
    return True;
}


csc* array_to_csc(c_int m, c_int n, double **A){
    c_int i,j, A_nnz, p_counter, num_vals, first_ele;
    c_int *A_p, *A_i;
    double *A_x;
    num_vals = 0;
    p_counter = 0;
    csc *final;
    A_nnz = 0;      
    if((final = c_malloc(sizeof(csc))) < 0)
        perror("MEM ERROR");
    for(j=0;j!=n;j++)
        for(i=0;i!=m;i++)
            if(A[i][j] != 0) A_nnz++;
    A_x = malloc(A_nnz * sizeof(c_float));
    A_i = malloc(A_nnz * sizeof(c_int));
    A_p = malloc((n+1)*sizeof(c_int));

    for(j=0;j!=n;j++){

        first_ele = 0;
        for(i=0;i!=m;i++){
            if(A[i][j] != 0){
                A_x[num_vals] = A[i][j];
                A_i[num_vals] = i;
                if(first_ele == 0){
                    A_p[p_counter] = num_vals;
                    p_counter++;
                    first_ele ++;
                }
                num_vals++;
            }
            
        }
    }
    A_p[n] = A_nnz;
    /*
    for(i=0;i<num_vals;i++)
        printf("%f ",A_x[i]);
    printf("\n");
    for(i=0;i<num_vals;i++)
        printf("%lld ",A_i[i]);
    printf("\n");
    for(i=0;i<p_counter+1;i++)
        printf("%lld ",A_p[i]);
    printf("\n");*/
    final->nzmax = A_nnz;
    final->m = m;
    final->n = n;
    final->p = malloc((n+1)*sizeof(c_int));
    final->p = A_p;
    final->i = malloc(A_nnz * sizeof(c_int));
    final->i = A_i;
    final->x = malloc(A_nnz * sizeof(c_float));
    final->x = A_x;
    final->nz = -1;
    return final;
        
}

double* higher_dim_pred(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length,
                        double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr, double* init_A){
    OSQPData* data;
    OSQPSettings  *settings;
    OSQPWorkspace *work;
    int i, j, k, in_set; 
    double temp, *b;
    double *x;
    double* results;
    results = malloc(length*sizeof(double));
    b = malloc(m*sizeof(double));
    x = malloc(trace_size*sizeof(double));
    if((data = (OSQPData *)c_malloc(sizeof(OSQPData))) < 0)
        perror("MEM ERROR");
    
    if((settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings))) < 0)
        perror("MEM ERROR");
    /*
    k = 0;
    for(j=0;j!=m*n;j++){
        
        printf("%lf ",init_A[j]);
        k++ ;
        if(k == n){
            k = 0;
            printf("\n");
        }
    }*/
    long current_time_step;
    if (settings) {
            osqp_set_default_settings(settings);
            settings->verbose = false;
    }
    /*
    printf("%f %f\n",P_data[0],P_data[1]);
    printf("%ld %ld\n",P_indices[0],P_indices[1]);
    printf("%ld %ld %ld\n",P_indptr[0],P_indptr[1],P_indptr[2]);
    printf("%f %f\n",A_data[0],A_data[1]);
    printf("%ld %ld\n",A_indices[0],A_indices[1]);
    printf("%ld %ld %ld\n",A_indptr[0],A_indptr[1],A_indptr[2]);*/
    
    if(data){
        data->n = n;
        data->m = m;
        data->P = csc_matrix(data->n, data->n, P_nnz, P_data, P_indices, P_indptr);
        data->q = q;
        data->A = csc_matrix(data->m, data->n, A_nnz, A_data, A_indices, A_indptr);
    }
    for(current_time_step = 0; current_time_step < length; current_time_step++){
        in_set = 0;
        if (data) {
            data->l = l;
            data->u = b;
            //printf("Timestamp: %ld and trace: ", current_time_step);
            for(j=0;j!=trace_size;j++){
                x[j] = traces[j][current_time_step];
                //printf("%f ",x[j]);
            }
            //printf(":");
            
            for(j=0;j!=m;j++){
                temp = 0;
                for(k=0;k!=trace_size;k++){
                    temp += init_A[j*n + k] * x[k];
                }
                // trace is in good set
                if(temp > u[j])  
                    in_set = 1;
                b[j] = u[j] - temp;
            }
            if(in_set == 0){
                for(i=0;i!=m;i++){
                    l[i] = b[i];
                    b[i] = INFINITY;
                }
            }else{
                for(i=0;i!=m;i++){
                    l[i] = -INFINITY;
                }
            }
            osqp_setup(&work, data, settings);
            osqp_solve(work);
            results[current_time_step] = sqrt(work->info->obj_val);
            if(in_set==0)
                results[current_time_step] = (double)(-1.0)*results[current_time_step];
        }
        
       
   }
   if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    
    if (settings) 
        c_free(settings);
    return results;
}
double* higher_dim_pred_threaded(long trace_size, long long int n, long long int m, double* q,double* l, double* u, int A_nnz, int P_nnz, double** traces, long length,
                        double* P_data, long long int* P_indices, long long int* P_indptr, double* A_data, long long int* A_indices, long long int* A_indptr, double* init_A){
    OSQPData* data;
    OSQPSettings  *settings;
    OSQPWorkspace *work;
    int i, j, k, in_set; 
    double temp, *b;
    double *x;
    double* results;
    results = malloc(length*sizeof(double));
    b = malloc(m*sizeof(double));
    x = malloc(trace_size*sizeof(double));
    if((data = (OSQPData *)c_malloc(sizeof(OSQPData))) < 0)
        perror("MEM ERROR");
    
    if((settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings))) < 0)
        perror("MEM ERROR");
    /*
    k = 0;
    for(j=0;j!=m*n;j++){
        
        printf("%lf ",init_A[j]);
        k++ ;
        if(k == n){
            k = 0;
            printf("\n");
        }
    }*/
    long current_time_step;
    if (settings) {
            osqp_set_default_settings(settings);
            settings->verbose = false;
    }
    /*
    printf("%f %f\n",P_data[0],P_data[1]);
    printf("%ld %ld\n",P_indices[0],P_indices[1]);
    printf("%ld %ld %ld\n",P_indptr[0],P_indptr[1],P_indptr[2]);
    printf("%f %f\n",A_data[0],A_data[1]);
    printf("%ld %ld\n",A_indices[0],A_indices[1]);
    printf("%ld %ld %ld\n",A_indptr[0],A_indptr[1],A_indptr[2]);*/
    clock_t begin = clock();
    for(current_time_step = 0; current_time_step < length; current_time_step++){
        in_set = 0;
        if (data) {
            data->n = n;
            data->m = m;
            data->P = csc_matrix(data->n, data->n, P_nnz, P_data, P_indices, P_indptr);
            data->q = q;
            data->A = csc_matrix(data->m, data->n, A_nnz, A_data, A_indices, A_indptr);
            data->l = l;
            data->u = b;
            //printf("Timestamp: %ld and trace: ", current_time_step);
            for(j=0;j!=trace_size;j++){
                x[j] = traces[j][current_time_step];
                //printf("%f ",x[j]);
            }
            //printf(":");
            
            for(j=0;j!=m;j++){
                temp = 0;
                for(k=0;k!=trace_size;k++){
                    temp += init_A[j*n + k] * x[k];
                }
                // trace is in good set
                if(temp > u[j])  
                    in_set = 1;
                b[j] = u[j] - temp;
            }
            if(in_set == 0){
                for(i=0;i!=m;i++){
                    l[i] = b[i];
                    b[i] = INFINITY;
                }
            }else{
                for(i=0;i!=m;i++){
                    l[i] = -INFINITY;
                }
            }
            osqp_setup(&work, data, settings);
            osqp_solve(work);
            results[current_time_step] = (double)(1.0)*sqrt(work->info->obj_val);
            if(in_set==0)
                results[current_time_step] = (double)(-1.0)*results[current_time_step];
        }
        
       
   }
   double time_spent = 0;
   clock_t end = clock();
   time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
   printf("C time higher dim time: %lf\n",time_spent);
   if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    
    if (settings) 
        c_free(settings);
    return results;
}

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
        //#pragma omp parallel
        //#pragma omp taskloop num_tasks(32)
        #pragma omp parallel for
        for(i = length - 1; i >= 0; i--){
            if(*(robustness + i) > max){
            max = *(robustness + i);
            }
            *(finally_robustness + i) = max;
        
        }
    }
    else{
        long current_time_step;
        #pragma omp parallel for
        for(current_time_step = length - 1; current_time_step >= 0; current_time_step--){
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
        #pragma omp parallel
        #pragma omp single
        #pragma omp taskloop num_tasks(32)
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
        #pragma omp parallel
        #pragma omp single
        #pragma omp taskloop num_tasks(32)
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
    /*
    double *left_traces = (double*)malloc(3*sizeof(double));
    double *right_traces = (double*)malloc(3*sizeof(double));
    left_traces[0] = 0;
    left_traces[1] = -1;
    left_traces[2] = 0;
    right_traces[0] = 0;
    right_traces[1] = 0;
    right_traces[2] = 0;
    printf("%d\n",in_set(left_traces,right_traces,3));
    */
    
    long length = 100000000;
    double time_spent = 0;
    float *left_traces = (float*)malloc(length*sizeof(float));
    float *right_traces = (float*)malloc(length*sizeof(float));
    float *time_stamps = (float*)malloc(length*sizeof(float));
    float *results = (float*)malloc(length*sizeof(float));
    
    long i;
    for(i = 0; i < length;i++){
        left_traces[i] = 1;
        right_traces[i] = 2;
        time_stamps[i] = i;
    }
    /*
    double time_spent = 0;
    c_float q[2] = {1.0, 1.0, };
    c_float l[3] = {1.0, 0.0, 0.0, };
    c_float u[3] = {1.0, 0.7, 0.7, };
    c_float **disp, **P;
    c_float test;
    //csc *test;
    disp = malloc(3 * sizeof(c_float*));
    P = malloc(2 * sizeof(c_float*));
    for (i=0; i<3; i++)
        disp[i] = malloc(2 * sizeof(c_float));
    for (i=0; i<2; i++)
        P[i] = malloc(2*sizeof(c_float));
    disp[0][0] = 1.0;
    disp[1][0] = 1.0;
    disp[2][0] = 0.0;
    disp[0][1] = 1.0;
    disp[1][1] = 0.0;
    disp[2][1] = 1.0;
    
    P[0][0] = 4.0;
    P[0][1] = 1.0;
    P[1][0] = 0.0;
    P[1][1] = 2.0;
    //higher_dim_pred(2, 3, q, l, u, disp, P,left_traces,length);
    //printf("%f\n",test);*/
/*
    if((test = (csc *)c_malloc(sizeof(csc))) < 0)
        perror("Error allocating memory");
    test = array_to_csc(3,2,disp);
    */
/*
    for(i = 0; i < length;i++){
        left_traces[i] = 1;
        right_traces[i] = 2;
        time_stamps[i] = i;
    }
    * */
    clock_t begin = clock();
    //predicate_setup(traces, 2.0f, 0.0f,length);
    results = c_finally_threaded(0.0f,100.0f, left_traces,time_stamps, length);
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%g\n",time_spent);
    //results[0] = 3.0f;


    printf("%f\n", results[1]);
}

