#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


float* eye(long length){
    int i;
    float* mat = (float*)calloc(length,sizeof(float));
    for(i=0; i<length; i++){
        *(mat + (i * length + i)) = 1;
    }
    return mat;
}

float* matmul(float* left_mat, long left_row, long left_col, float* right_mat, long right_row, long right_col){
    if(left_col != right_row){
        perror("Matrix dim mismatch for multiplication");
        exit(EXIT_FAILURE);
    }
    float* result = (float*)malloc(left_row * right_col * sizeof(float));
    long i,j,k;
    float cell_sum;
   
    for(i = 0; i < left_row; i++){
        for(j = 0; j < right_col; j++){
            cell_sum = 0;
            for(k = 0; k < left_col; k++){
                cell_sum += *(left_mat + (i * left_col + k)) * *(right_mat + (k * right_col + j));
            }
            printf("cell: %lf\n", cell_sum);
            *(result + (i * right_col + j)) = cell_sum;
        }
    }
   
    return result;
}

void matsub(float* left_mat, long left_row, long left_col, float* right_mat, long right_row, long right_col){
    if(left_col != right_col || left_row != right_row){
        perror("Matrix dim mismatch for subtraction");
        exit(EXIT_FAILURE);
    }
    
    long i, j;
    for(i = 0; i < left_row; i++){
        for(j = 0; j < left_col; j++){
            *(left_mat + (i * left_col + j)) = *(left_mat + (i * left_col + j)) - *(right_mat + (i * right_col + j));
        }
    }
}

void matadd(float* left_mat, long left_row, long left_col, float* right_mat, long right_row, long right_col){
    if(left_col != right_col || left_row != right_row){
        perror("Matrix dim mismatch for subtraction");
        exit(EXIT_FAILURE);
    }
    
    long i, j;
    for(i = 0; i < left_row; i++){
        for(j = 0; j < left_col; j++){
            *(left_mat + (i * left_col + j)) = *(left_mat + (i * left_col + j)) + *(right_mat + (i * right_col + j));
        }
    }
}

void matscaler(float scaler, float* mat, long row, long col){
    long i, j;
    for(i = 0; i < row; i++){
        for(j = 0; j < col; j++){
            *(mat + (i * col + j)) = scaler * *(mat + (i * col + j));
        }
    }
}

_Bool mat_lessthan_equal_to(float* left_mat, long left_row, long left_col, float* right_mat,long right_row, long right_col){
    if(left_col != right_col || left_row != right_row){
        perror("Matrix dim mismatch for lessthan");
        exit(EXIT_FAILURE);
    }
    
    long i,j;
    for(i = 0; i < left_row; i++){
        for(j = 0; j < left_col; j++){
            if(*(left_mat + (i * left_col + j)) > *(right_mat + (i * right_col + j))){
                return false;
            }
        }
    }

return true;
}

float* transpose(float* mat, long mat_row, long mat_col){
    float* transposed_mat = (float*)malloc(mat_row*mat_col*sizeof(float));
    long i,j; 
    for(i=0;i<mat_row;i++){
        for(j=0;j<mat_col;j++){
            *(transposed_mat + (j * mat_col + i)) = *(mat + (i * mat_col + j));
        }
    } 
    return transposed_mat;
}

//Qx - c
float* compute_gradient(float* Q, long Q_row, long Q_col, float* x, long x_row, long x_col, float* c, long c_row, long c_col){
    float* Qx = matmul(Q,Q_row,Q_col,x,x_row,x_col);
    matsub(Qx,Q_row,x_col,c,c_row,c_col);
    return Qx;
}
//(1/2)(x^t)Qx + (c^t)x
float compute_quadratic(float* Q, long Q_row,float* x, float* c){
    float* x_transpose = transpose(x, Q_row, 1);
    float* c_transpose = transpose(c,Q_row,1);
    
    matscaler(.5,x_transpose,Q_row,1);
    
    float* xtQ = matmul(x_transpose,1,Q_row,Q,Q_row,Q_row);
    float* xtQx = matmul(xtQ,1,Q_row,x,Q_row,1);
    
    float* ctx = matmul(c_transpose,1,Q_row,x,Q_row,1);
    
    float value = *xtQx + *ctx;
    free(x_transpose);
    free(c_transpose);
    free(xtQ);
    free(xtQx);
    free(ctx);
    
    return value;
}

float optimize(float* A, long A_row, long A_col, float* b, float* trace, float tolerance){
    float* x = malloc(A_col*sizeof(float));
    float* c = calloc(A_col,sizeof(float));
    float* Q = eye(A_col);
    
    matscaler(2,Q,A_col,A_col);
    
    
    float quad_value = compute_quadratic(Q,A_col,x,c);
   
    free(x);
    
    return quad_value;
}

int main(){
    float* A = (float*)malloc(2 * 3 * sizeof(float));
    float* B = (float*)malloc (3 * 2 * sizeof(float));
   
    *(A + (0 * 3 + 0)) = 1.0;
    *(A + (0 * 3 + 1)) = 1.0;
    *(A + (0 * 3 + 2)) = 1.0;
    *(A + (1 * 3 + 0)) = 1.0;
    *(A + (1 * 3 + 1)) = 1.0;
    *(A + (1 * 3 + 2)) = 1.0;

    *(B + (0 * 2 + 0)) = 2.0;
    *(B + (1 * 2 + 0)) = 1.0;
    *(B + (2 * 2 + 0)) = 4.0;
    *(B + (0 * 2 + 1)) = 2.0;
    *(B + (1 * 2 + 1)) = 1.0;
    *(B + (2 * 2 + 1)) = 4.0;

    printf("%d\n",mat_lessthan_equal_to(B,3,2,A,3,2));
   
    //long i,j;
    //for(i = 0; i < 3; i++){
        //for(j = 0; j < 2; j++){
            //printf("%lf, ",*(B + (i * 2 + j)));
        //}
        //printf("\n");
    //}
}
