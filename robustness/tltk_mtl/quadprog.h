#ifndef quadprog_H
#define quadprog_H

#include "f2c.h" 

int qpgen2_(doublereal *dmat, doublereal *dvec, integer *
    fddmat, integer *n, doublereal *sol, doublereal *lagr, doublereal *
    crval, doublereal *amat, doublereal *bvec, integer *fdamat, integer *
    q, integer *meq, integer *iact, integer *nact, integer *iter, 
    doublereal *work, integer *ierr)
{ 
