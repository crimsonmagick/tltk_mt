#include "f2c.h" 

int qpgen2_(doublereal *dmat, doublereal *dvec, integer *
    fddmat, integer *n, doublereal *sol, doublereal *lagr, doublereal *
    crval, doublereal *amat, doublereal *bvec, integer *fdamat, integer *
    q, integer *meq, integer *iact, integer *nact, integer *iter, 
    doublereal *work, integer *ierr);
    
    
void wrap_polyhedron_two(double** traces,double* C_f,double *b,double* results,int m_in , int n_in,long length);
void c_pred_bool(double* traces,double* C_f,double *b,double* results,int m , int n,long length);
