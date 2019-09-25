// example: construct a quadratic program from given iterators
// the QP below is the first quadratic program example in the user manual
#include <iostream>
#include <stdio.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>
// choose exact integral type
#ifdef CGAL_USE_GMP
#include <CGAL/Gmpz.h>
typedef CGAL::Gmpz ET;
#else
#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;
#endif

using namespace std;
// program and solution types
typedef CGAL::Quadratic_program_from_iterators
<int**,                                                // for A
 int*,                                                 // for b
 CGAL::Const_oneset_iterator<CGAL::Comparison_result>, // for r
 bool*,                                                // for fl
 int*,                                                 // for l
 bool*,                                                // for fu
 int*,                                                 // for u
 int**,                                                // for D
 int*>                                                 // for c 
Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

__global__ void quad(Program prog, Solution* sol){
    sol = CGAL::solve_quadratic_program(prog, ET());
}

int main() {
    int  Ax[] = {1, -1};                        // column for x
    int  Ay[] = {1,  2};                        // column for y
    int*  A[] = {Ax, Ay};                       // A comes columnwise
    int   b[] = {7, 4};                         // right-hand side
    CGAL::Const_oneset_iterator<CGAL::Comparison_result> 
        r(    CGAL::SMALLER);                 // constraints are "<="
    bool fl[] = {true, true};                   // both x, y are lower-bounded
    int   l[] = {0, 0};
    bool fu[] = {false, true};                  // only y is upper-bounded
    int   u[] = {0, 4};                         // x's u-entry is ignored
    int  D1[] = {2};                            // 2D_{1,1}
    int  D2[] = {0, 8};                         // 2D_{2,1}, 2D_{2,2}
    int*  D[] = {D1, D2};                       // D-entries on/below diagonal
    int   c[] = {0, -32};
    int  c0   = 64;                             // constant term
    // now construct the quadratic program; the first two parameters are
    // the number of variables and the number of constraints (rows of A)
    Program qp (2, 2, A, b, r, fl, l, fu, u, D, c, c0);
    // solve the program, using ET as the exact type
    //Solution* sol = (Solution*)malloc(sizeof(Solution));
    Solution* gpu_sol;
    Solution sol1 = Solution();
    Solution gpu_sol1 = Solution();
    Solution* sol = &sol1;
    if(cudaMalloc(&gpu_sol, sizeof(Solution)) != cudaSuccess)
        perror("GPU MEM ERROR");
    
    cudaMemcpy(gpu_sol, sol, sizeof(Solution), cudaMemcpyHostToDevice);
    quad<<<1,1>>>(qp,gpu_sol);
    
    if(cudaMemcpy(sol, gpu_sol, sizeof(Solution), cudaMemcpyDeviceToHost) != cudaSuccess)
        perror("GPU COPY ERROR");

    // output solution
    cout << (float)to_double(sol.objective_value()) << "\n";

    return 0;
}
