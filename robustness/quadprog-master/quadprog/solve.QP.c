/* solve.QP.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/sysinfo.h>
#include <unistd.h>

const int true  = 0;
const int false = 1;

/*  Copyright (C) 1995-2010 Berwin A. Turlach <Berwin.Turlach@gmail.com> */

/*  This program is free software; you can redistribute it and/or modify */
/*  it under the terms of the GNU General Public License as published by */
/*  the Free Software Foundation; either version 2 of the License, or */
/*  (at your option) any later version. */

/*  This program is distributed in the hope that it will be useful, */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/*  GNU General Public License for more details. */

/*  You should have received a copy of the GNU General Public License */
/*  along with this program; if not, write to the Free Software */
/*  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, */
/*  USA. */

/*  this routine uses the Goldfarb/Idnani algorithm to solve the */
/*  following minimization problem: */

/*        minimize  -d^T x + 1/2 *  x^T D x */
/*        where   A1^T x  = b1 */
/*                A2^T x >= b2 */

/*  the matrix D is assumed to be positive definite.  Especially, */
/*  w.l.o.g. D is assumed to be symmetric. */

/*  Input parameter: */
/*  dmat   nxn matrix, the matrix D from above (dp) */
/*         *** WILL BE DESTROYED ON EXIT *** */
/*         The user has two possibilities: */
/*         a) Give D (ierr=0), in this case we use routines from LINPACK */
/*            to decompose D. */
/*         b) To get the algorithm started we need R^-1, where D=R^TR. */
/*            So if it is cheaper to calculate R^-1 in another way (D may */
/*            be a band matrix) then with the general routine, the user */
/*            may pass R^{-1}.  Indicated by ierr not equal to zero. */
/*  dvec   nx1 vector, the vector d from above (dp) */
/*         *** WILL BE DESTROYED ON EXIT *** */
/*         contains on exit the solution to the initial, i.e., */
/*         unconstrained problem */
/*  fddmat scalar, the leading dimension of the matrix dmat */
/*  n      the dimension of dmat and dvec (int) */
/*  amat   nxq matrix, the matrix A from above (dp) [ A=(A1 A2)^T ] */
/*         *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE */
/*             CHANGED SIGNES ON EXIT *** */
/*  bvec   qx1 vector, the vector of constants b in the constraints (dp) */
/*         [ b = (b1^T b2^T)^T ] */
/*         *** ENTRIES CORRESPONDING TO EQUALITY CONSTRAINTS MAY HAVE */
/*             CHANGED SIGNES ON EXIT *** */
/*  fdamat the first dimension of amat as declared in the calling program. */
/*         fdamat >= n !! */
/*  q      integer, the number of constraints. */
/*  meq    integer, the number of equality constraints, 0 <= meq <= q. */
/*  ierr   integer, code for the status of the matrix D: */
/*            ierr =  0, we have to decompose D */
/*            ierr != 0, D is already decomposed into D=R^TR and we were */
/*                       given R^{-1}. */

/*  Output parameter: */
/*  sol   nx1 the final solution (x in the notation above) */
/*  lagr  qx1 the final Lagrange multipliers */
/*  crval scalar, the value of the criterion at the minimum */
/*  iact  qx1 vector, the constraints which are active in the final */
/*        fit (int) */
/*  nact  scalar, the number of constraints active in the final fit (int) */
/*  iter  2x1 vector, first component gives the number of "main" */
/*        iterations, the second one says how many constraints were */
/*        deleted after they became active */
/*  ierr  integer, error code on exit, if */
/*           ierr = 0, no problems */
/*           ierr = 1, the minimization problem has no solution */
/*           ierr = 2, problems with decomposing D, in this case sol */
/*                     contains garbage!! */

/*  Working space: */
/*  work  vector with length at least 2*n+r*(r+5)/2 + 2*q +1 */
/*        where r=min(n,q) */

/* Subroutine */ int qpgen2_(doublereal *dmat, doublereal *dvec, integer *
	fddmat, integer *n, doublereal *sol, doublereal *lagr, doublereal *
	crval, doublereal *amat, doublereal *bvec, integer *fdamat, integer *
	q, integer *meq, integer *iact, integer *nact, integer *iter, 
	doublereal *work, integer *ierr)
{
    /* System generated locals */
    integer dmat_dim1, dmat_offset, amat_dim1, amat_offset, i__1, i__2;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double sqrt(doublereal), d_sign(doublereal *, doublereal *);

    /* Local variables */
    static integer i__, j, l, r__, l1;
    static doublereal t1, gc, gs, nu, tt;
    static integer it1, nvl;
    static doublereal sum;
    static integer info;
    static doublereal tmpa, tmpb, temp;
    static integer iwrm, iwrv, iwsv, iwuv, iwzv;
    static logical t1inf, t2min;
    extern /* Subroutine */ int dpofa_(doublereal *, integer *, integer *, 
	    integer *), dpori_(doublereal *, integer *, integer *), dposl_(
	    doublereal *, integer *, integer *, doublereal *);
    static integer iwnbv;
    static doublereal vsmall;

    /* Parameter adjustments */
    --dvec;
    dmat_dim1 = *fddmat;
    dmat_offset = 1 + dmat_dim1;
    dmat -= dmat_offset;
    --sol;
    --lagr;
    --bvec;
    amat_dim1 = *fdamat;
    amat_offset = 1 + amat_dim1;
    amat -= amat_offset;
    --iact;
    --iter;
    --work;

    /* Function Body */
    r__ = min(*n,*q);
    l = (*n << 1) + r__ * (r__ + 5) / 2 + (*q << 1) + 1;

/*     code gleaned from Powell's ZQPCVX routine to determine a small */
/*     number  that can be assumed to be an upper bound on the relative */
/*     precision of the computer arithmetic. */

    vsmall = 1e-60;
L1:
    vsmall += vsmall;
    tmpa = vsmall * .1 + 1.;
    tmpb = vsmall * .2 + 1.;
    if (tmpa <= 1.) {
	goto L1;
    }
    if (tmpb <= 1.) {
	goto L1;
    }

/* store the initial dvec to calculate below the unconstrained minima of */
/* the critical value. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = dvec[i__];
/* L10: */
    }
    i__1 = l;
    for (i__ = *n + 1; i__ <= i__1; ++i__) {
	work[i__] = 0.;
/* L11: */
    }
    i__1 = *q;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iact[i__] = 0;
	lagr[i__] = 0.;
/* L12: */
    }

/* get the initial solution */

    if (*ierr == 0) {
	dpofa_(&dmat[dmat_offset], fddmat, n, &info);
	if (info != 0) {
	    *ierr = 2;
	    goto L999;
	}
	dposl_(&dmat[dmat_offset], fddmat, n, &dvec[1]);
	dpori_(&dmat[dmat_offset], fddmat, n);
    } else {

/* Matrix D is already factorized, so we have to multiply d first with */
/* R^-T and then with R^-1.  R^-1 is stored in the upper half of the */
/* array dmat. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sol[j] = 0.;
	    i__2 = j;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sol[j] += dmat[i__ + j * dmat_dim1] * dvec[i__];
/* L21: */
	    }
/* L20: */
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    dvec[j] = 0.;
	    i__2 = *n;
	    for (i__ = j; i__ <= i__2; ++i__) {
		dvec[j] += dmat[j + i__ * dmat_dim1] * sol[i__];
/* L23: */
	    }
/* L22: */
	}
    }

/* set lower triangular of dmat to zero, store dvec in sol and */
/* calculate value of the criterion at unconstrained minima */

    *crval = 0.;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	sol[j] = dvec[j];
	*crval += work[j] * sol[j];
	work[j] = 0.;
	i__2 = *n;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    dmat[i__ + j * dmat_dim1] = 0.;
/* L32: */
	}
/* L30: */
    }
    *crval = -(*crval) / 2.;
    *ierr = 0;

/* calculate some constants, i.e., from which index on the different */
/* quantities are stored in the work matrix */

    iwzv = *n;
    iwrv = iwzv + *n;
    iwuv = iwrv + r__;
    iwrm = iwuv + r__ + 1;
    iwsv = iwrm + r__ * (r__ + 1) / 2;
    iwnbv = iwsv + *q;

/* calculate the norm of each column of the A matrix */

    i__1 = *q;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sum = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    sum += amat[j + i__ * amat_dim1] * amat[j + i__ * amat_dim1];
/* L52: */
	}
	work[iwnbv + i__] = sqrt(sum);
/* L51: */
    }
    *nact = 0;
    iter[1] = 0;
    iter[2] = 0;
L50:

/* start a new iteration */

    ++iter[1];

/* calculate all constraints and check which are still violated */
/* for the equality constraints we have to check whether the normal */
/* vector has to be negated (as well as bvec in that case) */

    l = iwsv;
    i__1 = *q;
    for (i__ = 1; i__ <= i__1; ++i__) {
	++l;
	sum = -bvec[i__];
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    sum += amat[j + i__ * amat_dim1] * sol[j];
/* L61: */
	}
	if (abs(sum) < vsmall) {
	    sum = 0.;
	}
	if (i__ > *meq) {
	    work[l] = sum;
	} else {
	    work[l] = -abs(sum);
	    if (sum > 0.) {
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    amat[j + i__ * amat_dim1] = -amat[j + i__ * amat_dim1];
/* L62: */
		}
		bvec[i__] = -bvec[i__];
	    }
	}
/* L60: */
    }

/* as safeguard against rounding errors set already active constraints */
/* explicitly to zero */

    i__1 = *nact;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[iwsv + iact[i__]] = 0.;
/* L70: */
    }

/* we weight each violation by the number of non-zero elements in the */
/* corresponding row of A. then we choose the violated constraint which */
/* has maximal absolute value, i.e., the minimum. */
/* by obvious commenting and uncommenting we can choose the strategy to */
/* take always the first constraint which is violated. ;-) */

    nvl = 0;
    temp = 0.;
    i__1 = *q;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (work[iwsv + i__] < temp * work[iwnbv + i__]) {
	    nvl = i__;
	    temp = work[iwsv + i__] / work[iwnbv + i__];
	}
/*         if (work(iwsv+i) .LT. 0.d0) then */
/*            nvl = i */
/*            goto 72 */
/*         endif */
/* L71: */
    }
/* L72: */
    if (nvl == 0) {
	i__1 = *nact;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    lagr[iact[i__]] = work[iwuv + i__];
/* L73: */
	}
	goto L999;
    }

/* calculate d=J^Tn^+ where n^+ is the normal vector of the violated */
/* constraint. J is stored in dmat in this implementation!! */
/* if we drop a constraint, we have to jump back here. */

L55:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	sum = 0.;
	i__2 = *n;
	for (j = 1; j <= i__2; ++j) {
	    sum += dmat[j + i__ * dmat_dim1] * amat[j + nvl * amat_dim1];
/* L81: */
	}
	work[i__] = sum;
/* L80: */
    }

/* Now calculate z = J_2 d_2 */

    l1 = iwzv;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[l1 + i__] = 0.;
/* L90: */
    }
    i__1 = *n;
    for (j = *nact + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[l1 + i__] += dmat[i__ + j * dmat_dim1] * work[j];
/* L93: */
	}
/* L92: */
    }

/* and r = R^{-1} d_1, check also if r has positive elements (among the */
/* entries corresponding to inequalities constraints). */

    t1inf = TRUE_;
    for (i__ = *nact; i__ >= 1; --i__) {
	sum = work[i__];
	l = iwrm + i__ * (i__ + 3) / 2;
	l1 = l - i__;
	i__1 = *nact;
	for (j = i__ + 1; j <= i__1; ++j) {
	    sum -= work[l] * work[iwrv + j];
	    l += j;
/* L96: */
	}
	sum /= work[l1];
	work[iwrv + i__] = sum;
	if (iact[i__] <= *meq) {
	    goto L95;
	}
	if (sum <= 0.) {
	    goto L95;
	}
/* L7: */
	t1inf = FALSE_;
	it1 = i__;
L95:
	;
    }

/* if r has positive elements, find the partial step length t1, which is */
/* the maximum step in dual space without violating dual feasibility. */
/* it1  stores in which component t1, the min of u/r, occurs. */

    if (! t1inf) {
	t1 = work[iwuv + it1] / work[iwrv + it1];
	i__1 = *nact;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (iact[i__] <= *meq) {
		goto L100;
	    }
	    if (work[iwrv + i__] <= 0.) {
		goto L100;
	    }
	    temp = work[iwuv + i__] / work[iwrv + i__];
	    if (temp < t1) {
		t1 = temp;
		it1 = i__;
	    }
L100:
	    ;
	}
    }

/* test if the z vector is equal to zero */

    sum = 0.;
    i__1 = iwzv + *n;
    for (i__ = iwzv + 1; i__ <= i__1; ++i__) {
	sum += work[i__] * work[i__];
/* L110: */
    }
    if (abs(sum) <= vsmall) {

/* No step in primal space such that the new constraint becomes */
/* feasible. Take step in dual space and drop a constant. */

	if (t1inf) {

/* No step in dual space possible either, problem is not solvable */

	    *ierr = 1;
	    goto L999;
	} else {

/* we take a partial step in dual space and drop constraint it1, */
/* that is, we drop the it1-th active constraint. */
/* then we continue at step 2(a) (marked by label 55) */

	    i__1 = *nact;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[iwuv + i__] -= t1 * work[iwrv + i__];
/* L111: */
	    }
	    work[iwuv + *nact + 1] += t1;
	    goto L700;
	}
    } else {

/* compute full step length t2, minimum step in primal space such that */
/* the constraint becomes feasible. */
/* keep sum (which is z^Tn^+) to update crval below! */

	sum = 0.;
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sum += work[iwzv + i__] * amat[i__ + nvl * amat_dim1];
/* L120: */
	}
	tt = -work[iwsv + nvl] / sum;
	t2min = TRUE_;
	if (! t1inf) {
	    if (t1 < tt) {
		tt = t1;
		t2min = FALSE_;
	    }
	}

/* take step in primal and dual space */

	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sol[i__] += tt * work[iwzv + i__];
/* L130: */
	}
	*crval += tt * sum * (tt / 2. + work[iwuv + *nact + 1]);
	i__1 = *nact;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[iwuv + i__] -= tt * work[iwrv + i__];
/* L131: */
	}
	work[iwuv + *nact + 1] += tt;

/* if it was a full step, then we check wheter further constraints are */
/* violated otherwise we can drop the current constraint and iterate once */
/* more */
	if (t2min) {

/* we took a full step. Thus add constraint nvl to the list of active */
/* constraints and update J and R */

	    ++(*nact);
	    iact[*nact] = nvl;

/* to update R we have to put the first nact-1 components of the d vector */
/* into column (nact) of R */

	    l = iwrm + (*nact - 1) * *nact / 2 + 1;
	    i__1 = *nact - 1;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[l] = work[i__];
		++l;
/* L150: */
	    }

/* if now nact=n, then we just have to add the last element to the new */
/* row of R. */
/* Otherwise we use Givens transformations to turn the vector d(nact:n) */
/* into a multiple of the first unit vector. That multiple goes into the */
/* last element of the new row of R and J is accordingly updated by the */
/* Givens transformations. */

	    if (*nact == *n) {
		work[l] = work[*n];
	    } else {
		i__1 = *nact + 1;
		for (i__ = *n; i__ >= i__1; --i__) {

/* we have to find the Givens rotation which will reduce the element */
/* (l1) of d to zero. */
/* if it is already zero we don't have to do anything, except of */
/* decreasing l1 */

		    if (work[i__] == 0.) {
			goto L160;
		    }
/* Computing MAX */
		    d__3 = (d__1 = work[i__ - 1], abs(d__1)), d__4 = (d__2 = 
			    work[i__], abs(d__2));
		    gc = max(d__3,d__4);
/* Computing MIN */
		    d__3 = (d__1 = work[i__ - 1], abs(d__1)), d__4 = (d__2 = 
			    work[i__], abs(d__2));
		    gs = min(d__3,d__4);
		    d__1 = gc * sqrt(gs * gs / (gc * gc) + 1);
		    temp = d_sign(&d__1, &work[i__ - 1]);
		    gc = work[i__ - 1] / temp;
		    gs = work[i__] / temp;

/* The Givens rotation is done with the matrix (gc gs, gs -gc). */
/* If gc is one, then element (i) of d is zero compared with element */
/* (l1-1). Hence we don't have to do anything. */
/* If gc is zero, then we just have to switch column (i) and column (i-1) */
/* of J. Since we only switch columns in J, we have to be careful how we */
/* update d depending on the sign of gs. */
/* Otherwise we have to apply the Givens rotation to these columns. */
/* The i-1 element of d has to be updated to temp. */

		    if (gc == 1.) {
			goto L160;
		    }
		    if (gc == 0.) {
			work[i__ - 1] = gs * temp;
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    temp = dmat[j + (i__ - 1) * dmat_dim1];
			    dmat[j + (i__ - 1) * dmat_dim1] = dmat[j + i__ * 
				    dmat_dim1];
			    dmat[j + i__ * dmat_dim1] = temp;
/* L170: */
			}
		    } else {
			work[i__ - 1] = temp;
			nu = gs / (gc + 1.);
			i__2 = *n;
			for (j = 1; j <= i__2; ++j) {
			    temp = gc * dmat[j + (i__ - 1) * dmat_dim1] + gs *
				     dmat[j + i__ * dmat_dim1];
			    dmat[j + i__ * dmat_dim1] = nu * (dmat[j + (i__ - 
				    1) * dmat_dim1] + temp) - dmat[j + i__ * 
				    dmat_dim1];
			    dmat[j + (i__ - 1) * dmat_dim1] = temp;
/* L180: */
			}
		    }
L160:
		    ;
		}

/* l is still pointing to element (nact,nact) of the matrix R. */
/* So store d(nact) in R(nact,nact) */
		work[l] = work[*nact];
	    }
	} else {

/* we took a partial step in dual space. Thus drop constraint it1, */
/* that is, we drop the it1-th active constraint. */
/* then we continue at step 2(a) (marked by label 55) */
/* but since the fit changed, we have to recalculate now "how much" */
/* the fit violates the chosen constraint now. */

	    sum = -bvec[nvl];
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum += sol[j] * amat[j + nvl * amat_dim1];
/* L190: */
	    }
	    if (nvl > *meq) {
		work[iwsv + nvl] = sum;
	    } else {
		work[iwsv + nvl] = -abs(sum);
		if (sum > 0.) {
		    i__1 = *n;
		    for (j = 1; j <= i__1; ++j) {
			amat[j + nvl * amat_dim1] = -amat[j + nvl * amat_dim1]
				;
/* L191: */
		    }
		    bvec[nvl] = -bvec[nvl];
		}
	    }
	    goto L700;
	}
    }
    goto L50;

/* Drop constraint it1 */

L700:

/* if it1 = nact it is only necessary to update the vector u and nact */

    if (it1 == *nact) {
	goto L799;
    }

/* After updating one row of R (column of J) we will also come back here */

L797:

/* we have to find the Givens rotation which will reduce the element */
/* (it1+1,it1+1) of R to zero. */
/* if it is already zero we don't have to do anything except of updating */
/* u, iact, and shifting column (it1+1) of R to column (it1) */
/* l  will point to element (1,it1+1) of R */
/* l1 will point to element (it1+1,it1+1) of R */

    l = iwrm + it1 * (it1 + 1) / 2 + 1;
    l1 = l + it1;
    if (work[l1] == 0.) {
	goto L798;
    }
/* Computing MAX */
    d__3 = (d__1 = work[l1 - 1], abs(d__1)), d__4 = (d__2 = work[l1], abs(
	    d__2));
    gc = max(d__3,d__4);
/* Computing MIN */
    d__3 = (d__1 = work[l1 - 1], abs(d__1)), d__4 = (d__2 = work[l1], abs(
	    d__2));
    gs = min(d__3,d__4);
    d__1 = gc * sqrt(gs * gs / (gc * gc) + 1);
    temp = d_sign(&d__1, &work[l1 - 1]);
    gc = work[l1 - 1] / temp;
    gs = work[l1] / temp;

/* The Givens rotatin is done with the matrix (gc gs, gs -gc). */
/* If gc is one, then element (it1+1,it1+1) of R is zero compared with */
/* element (it1,it1+1). Hence we don't have to do anything. */
/* if gc is zero, then we just have to switch row (it1) and row (it1+1) */
/* of R and column (it1) and column (it1+1) of J. Since we swithc rows in */
/* R and columns in J, we can ignore the sign of gs. */
/* Otherwise we have to apply the Givens rotation to these rows/columns. */

    if (gc == 1.) {
	goto L798;
    }
    if (gc == 0.) {
	i__1 = *nact;
	for (i__ = it1 + 1; i__ <= i__1; ++i__) {
	    temp = work[l1 - 1];
	    work[l1 - 1] = work[l1];
	    work[l1] = temp;
	    l1 += i__;
/* L710: */
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = dmat[i__ + it1 * dmat_dim1];
	    dmat[i__ + it1 * dmat_dim1] = dmat[i__ + (it1 + 1) * dmat_dim1];
	    dmat[i__ + (it1 + 1) * dmat_dim1] = temp;
/* L711: */
	}
    } else {
	nu = gs / (gc + 1.);
	i__1 = *nact;
	for (i__ = it1 + 1; i__ <= i__1; ++i__) {
	    temp = gc * work[l1 - 1] + gs * work[l1];
	    work[l1] = nu * (work[l1 - 1] + temp) - work[l1];
	    work[l1 - 1] = temp;
	    l1 += i__;
/* L720: */
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = gc * dmat[i__ + it1 * dmat_dim1] + gs * dmat[i__ + (it1 + 
		    1) * dmat_dim1];
	    dmat[i__ + (it1 + 1) * dmat_dim1] = nu * (dmat[i__ + it1 * 
		    dmat_dim1] + temp) - dmat[i__ + (it1 + 1) * dmat_dim1];
	    dmat[i__ + it1 * dmat_dim1] = temp;
/* L721: */
	}
    }

/* shift column (it1+1) of R to column (it1) (that is, the first it1 */
/* elements). The posit1on of element (1,it1+1) of R was calculated above */
/* and stored in l. */

L798:
    l1 = l - it1;
    i__1 = it1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[l1] = work[l];
	++l;
	++l1;
/* L730: */
    }

/* update vector u and iact as necessary */
/* Continue with updating the matrices J and R */

    work[iwuv + it1] = work[iwuv + it1 + 1];
    iact[it1] = iact[it1 + 1];
    ++it1;
    if (it1 < *nact) {
	goto L797;
    }
L799:
    work[iwuv + *nact] = work[iwuv + *nact + 1];
    work[iwuv + *nact + 1] = 0.;
    iact[*nact] = 0;
    --(*nact);
    ++iter[2];
    goto L55;
L999:
    return 0;
} /* qpgen2_ */

//int min(int left, int right){
    //if(left < right){
        //return left;
    //}
    //return right;
//}

void transpose(double* mat, int mat_row, int mat_col,double* result){
    long i,j; 
    //double* result = (double*)calloc(mat_row*mat_col,sizeof(double));
    for(i=0;i<mat_col;i++){
        for(j=0;j<mat_row;j++){
            *(result + (j * mat_col + i)) = *(mat + (i * mat_row + j));
        }
    } 
}

void matmul(double* left_mat, int left_row, int left_col, double* right_mat, int right_row, int right_col, double* result){
    if(left_col != right_row){
        perror("Matrix dim mismatch for multiplication");
        exit(EXIT_FAILURE);
    }
    //double* result = (double*)malloc(left_row * right_col * sizeof(double));
    long i,j,k;
    double cell_sum;
   
    for(i = 0; i < left_row; i++){
        for(j = 0; j < right_col; j++){
            cell_sum = 0;
            for(k = 0; k < left_col; k++){
                cell_sum += *(left_mat + (i * left_col + k)) * *(right_mat + (k * right_col + j));
            }
            *(result + (i * right_col + j)) = cell_sum;
        }
    }
}

void matmulcol(double* left_mat, int left_row, int left_col, double* right_mat, int right_row, int right_col, double* result){
    if(left_col != right_row){
        perror("Matrix dim mismatch for multiplication");
        exit(EXIT_FAILURE);
    }
    //double* result = (double*)malloc(left_row * right_col * sizeof(double));
    long i,j,k;
    double cell_sum;
   
    for(i = 0; i < left_row; i++){
        for(j = 0; j < right_col; j++){
            cell_sum = 0;
            for(k = 0; k < left_col; k++){
                cell_sum += *(left_mat + (k * left_row + i)) * *(right_mat + (j * left_col + k));
            }
            *(result + (i * right_col + j)) = cell_sum;
        }
    }
}

void matsub(double* left_mat, int left_row, int left_col, double* right_mat, int right_row, int right_col){
    if(left_col != right_col || left_row != right_row){
        //perror("Matrix dim mismatch for subtraction");
        exit(EXIT_FAILURE);
    }
    
    long i;
    for(i = 0; i < left_row; i++){
            left_mat[i] = left_mat[i] - right_mat[i];
    }
}

double* eye(int length){
    int i;
    double* mat = (double*)calloc(length,sizeof(double));
    for(i=0; i<length; i++){
        *(mat + (i * length + i)) = 1;
    }
    return mat;
}
void matscaler(double scaler, double* mat, int row, int col){
    int i, j;
    for(i = 0; i < row; i++){
        for(j = 0; j < col; j++){
            *(mat + (j * row + i)) = scaler * *(mat + (j * row + i));
        }
    }
}


int matlessthaneq(double* left_mat, double* right_mat, int rows, int cols){
    int i;
    for(i=0;i < rows;i++){
        if(left_mat[i] > right_mat[i]){
            return false;
        }
    }
    return true;
}

int matgreaterthaneq(double* left_mat, double* right_mat, int rows, int cols){
    int i;
    for(i=0;i < rows;i++){
        if(left_mat[i] < right_mat[i]){
            return false;
        }
    }
    return true;
}


struct thread_package
{
    double* C;
    double* b;
    int n;
    int m;
    double** traces;
    long length;
    double* results;
    long start_index;
    long sub_length;
};

double* translate_c_fortran(double* mat, double* translated,int n, int m){
    int i,j;
    //double* translated = (double*)malloc(m * n * sizeof(double));
    for(i=0;i<n;i++){
        for(j=0;j<m;j++){
            *(translated + (i*m +j)) = *(mat + (j*n + i));
        }
    }
    free(mat);
    return translated;
}

int rowmajorindexfromcolumnmajorindex(int columnmajorindex, int width, int height) {
    int row = columnmajorindex % height;
    int column = columnmajorindex / height;
    return row * width + column;
}


int columnmajorindexfromrowmajorindex(int rowmajorindex, int width, int height) {
    return rowmajorindexfromcolumnmajorindex(rowmajorindex, height, width);
}

double* translate_fortran_c(double* mat,double* translated ,int n, int m){
    int i,j;
    //double* translated = (double*)malloc(m * n * sizeof(double));
    for(i=0;i<n;i++){
        for(j=0;j<m;j++){
            *(translated + i*m+j) = *(mat + (columnmajorindexfromrowmajorindex(i*n+j,n,m)));
        }
    }
    //free(mat);
    return translated;
}


double calc_projection(double* C,double* b,double* trace ,int m, int n, int current_row){
    //Calculates the projection of the trace onto a plane
    int i;
    double dot_product = 0;
    double norm = 0;
    for(i=0;i<n;i++){
        dot_product += (*(C +((i*m) + current_row))* *(trace + i));
        norm += (*(C +((i*m) + current_row)))*(*(C +((i*m) + current_row)));
    }
    dot_product -= *(b + current_row);
    return fabs(dot_product)/sqrt(norm);
    
}

double calc_depth(double* C, double* b, double* trace, int m, int n){
    int i;
    double min = INFINITY;
    double row_distance;
    for(i=0; i<m; i++){
        row_distance = calc_projection(C,b,trace,m,n,i);
        if(row_distance < min){
            min = row_distance;
        }
    }
    return min;
}

void wrap_polyhedron_thread_task(double** traces,double* C_f,double *b,double* results,int m_in , int n_in,long start, long stop, long length){
    double* a;
    int ierr;
    int meq;
    int* iters;
    int nact; 
    double* sol;
    double* lagr;
    double* work;
    int* iact; 
    double* b_sub;
    double* A_t_trace;
    double* G;
    double* C_temp;
    //double* results;
    double* C_t;
    double* C_f_temp;
    //printf("m:%d n:%d\n",m,n);
    
    int j;
    long i = 0;
    ierr = 1;
    meq = 0;
    nact = 0; 
    int m = m_in;
    int n = n_in;
    
    iters = (int*)malloc(2*sizeof(int));
    A_t_trace = (double*)malloc(m * 1 * sizeof(double));
    //results = (double*)malloc(length * sizeof(double));
    
    
    if(!(C_t = (double*)malloc(m*n*sizeof(double)))){
        perror("C init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(C_temp = (double*)malloc(m*n*sizeof(double)))){
        perror("C_temp init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(C_f_temp = (double*)malloc(m*n*sizeof(double)))){
        perror("C_f_temp init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(b_sub = (double*)malloc(m*sizeof(double)))){
        perror("b_sub init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(G = (double*)calloc(n*n,sizeof(double)))){
        perror("G init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(a = (double*)malloc(n*sizeof(double)))){
        perror("a init error");
        exit(EXIT_FAILURE);
    }
    
    sol = (double*)calloc(n*1,sizeof(double));
    lagr = (double*)calloc(m,sizeof(double));
    iact = (int*)calloc(m,sizeof(int));
    work = (double*)calloc(2*n+min(n, m)*(min(n, m)+5)/2 + 2*m +1,sizeof(double));
    
    //memcpy(C,C_f,m*n*sizeof(double));
    
    //translate_fortran_c(C_f,C,m,n);
    
    for(i = stop - 1; i >= start; i--){
        //refresh values incase qpgen destroyed them
        //printf("Time step: %ld\n",i);
        m = m_in;
        n = n_in;

        ierr = 1;
        meq = 0;
        nact = 0; 
        memset(iters, 0, 2*sizeof(int));
        memset(a,0,n*sizeof(double));
        memset(sol,0,n*sizeof(double));
        memset(lagr,0,m*sizeof(double));
        memset(iact,0,m*sizeof(int));
        memset(work,0,(2*n+min(n, m)*(min(n, m)+5)/2 + 2*m +1) * sizeof(double));
        memset(G, 0 , n*n*sizeof(double));    
        memset(A_t_trace,0,m*sizeof(double));
        
        memcpy(C_f_temp,C_f,m*n*sizeof(double));
        memcpy(b_sub,b,m*sizeof(double));
        
        
        
        //Fill G to be R^-1 n by n Matrix where 2*I = R^T * R
        for(j=0;j<n;j++){
             *(G + (j * n + j)) = 1;
        }
    
        //Multiply the current A by the current trace to see if we are calculating depth or distance
        matmulcol(C_f_temp,m,n,traces[i],n,1,A_t_trace);
        
        
        
        //Check if Ax >= b if it is multiply (b - Ax) and A by -1
        //if(matgreaterthaneq(A_t_trace,b_sub,m,1)){
                

        //}

        //Transpse A to be used with qpgen
        //for(w=0; w < m; w ++){
            //printf("A_t_trace: %lf || b: %lf\n",A_t_trace[w],b[w]);
            
        //}
        
        if(!matlessthaneq(A_t_trace,b,m,1)){
            results[i] = calc_depth(C_f,b,traces[0],m,n);
        }else{
            // Subtract A*x from b (b - A*x) 
            matsub(b_sub,m,1,A_t_trace,m,1);
            //Flip sign because qpgen works with Ax >= b
            matscaler(-1.0,b_sub,m,1);
            matscaler(-1.0,C_f_temp,m,n);
                
            //Transpse A to be used with qpgen
            transpose(C_f_temp,m,n , C_t);
            
            qpgen2_(G,a,&n,&n,sol,lagr,&results[i],C_t,b_sub,&n,&m,&meq,iact,&nact,iters,work,&ierr);
            results[i] = -1*sqrt(2*results[i]);
        }
        //free(traces[i]);
    }
    //printf("result: %f\n" ,results[0]);
    
    //printf("ierr: %d\n" ,ierr);
    //printf("DONE\n");
}

void wrap_polyhedron_threaded(double** traces,double* C_f,double *b,double* results,int m_in , int n_in,long length){
    long thread_count = sysconf(_SC_NPROCESSORS_ONLN);
    //long current_time_step;
    long division_length = length / thread_count;
    long overflow_work = length % thread_count;
    long task;
    //printf("task over_flow: %ld\n", overflow_work);
    #pragma omp parallel for num_threads(thread_count)
    for(task = 0; task < thread_count; task++){
        long start_index = task * division_length;
        long end_index = (task + 1) * division_length;
        
        if(task == (thread_count - 1)){
            end_index += overflow_work;
        }
        
        //finally_thread_task(start_index,end_index,lower_time_bound,upper_time_bound,robustness,time_stamps,finally_robustness,length
        wrap_polyhedron_thread_task(traces,C_f,b,results,m_in,n_in,start_index,end_index,length);
    }
}

void wrap_polyhedron_two(double** traces,double* C_f,double *b,double* results,int m_in , int n_in,long length){
    double* a;
    int ierr;
    int meq;
    int* iters;
    int nact; 
    double* sol;
    double* lagr;
    double* work;
    int* iact; 
    double* b_sub;
    double* A_t_trace;
    double* G;
    double* C_temp;
    //double* results;
    double* C_t;
    double* C_f_temp;
    //printf("m:%d n:%d\n",m,n);
    
    int j;
    long i = 0;
    ierr = 1;
    meq = 0;
    nact = 0; 
    int m = m_in;
    int n = n_in;
    
    iters = (int*)malloc(2*sizeof(int));
    A_t_trace = (double*)malloc(m * 1 * sizeof(double));
    //results = (double*)malloc(length * sizeof(double));
    
    
    if(!(C_t = (double*)malloc(m*n*sizeof(double)))){
        perror("C init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(C_temp = (double*)malloc(m*n*sizeof(double)))){
        perror("C_temp init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(C_f_temp = (double*)malloc(m*n*sizeof(double)))){
        perror("C_f_temp init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(b_sub = (double*)malloc(m*sizeof(double)))){
        perror("b_sub init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(G = (double*)calloc(n*n,sizeof(double)))){
        perror("G init error");
        exit(EXIT_FAILURE);
    }
    
    if(!(a = (double*)malloc(n*sizeof(double)))){
        perror("a init error");
        exit(EXIT_FAILURE);
    }
    
    sol = (double*)calloc(n*1,sizeof(double));
    lagr = (double*)calloc(m,sizeof(double));
    iact = (int*)calloc(m,sizeof(int));
    work = (double*)calloc(2*n+min(n, m)*(min(n, m)+5)/2 + 2*m +1,sizeof(double));
    
    //memcpy(C,C_f,m*n*sizeof(double));
    
    //translate_fortran_c(C_f,C,m,n);
    
    for(i = 0; i < length; i++){
        //refresh values incase qpgen destroyed them

        m = m_in;
        n = n_in;

        ierr = 1;
        meq = 0;
        nact = 0; 
        memset(iters, 0, 2*sizeof(int));
        memset(a,0,n*sizeof(double));
        memset(sol,0,n*sizeof(double));
        memset(lagr,0,m*sizeof(double));
        memset(iact,0,m*sizeof(int));
        memset(work,0,(2*n+min(n, m)*(min(n, m)+5)/2 + 2*m +1) * sizeof(double));
        memset(G, 0 , n*n*sizeof(double));    
        memset(A_t_trace,0,m*sizeof(double));
        
        memcpy(C_f_temp,C_f,m*n*sizeof(double));
        memcpy(b_sub,b,m*sizeof(double));
        
        
        
        //Fill G to be R^-1 n by n Matrix where 2*I = R^T * R
        for(j=0;j<n;j++){
             *(G + (j * n + j)) = 1;
        }
    
        //Multiply the current A by the current trace to see if we are calculating depth or distance
        matmulcol(C_f_temp,m,n,traces[i],n,1,A_t_trace);
        
        
        
        //Check if Ax >= b if it is multiply (b - Ax) and A by -1
        //if(matgreaterthaneq(A_t_trace,b_sub,m,1)){
                

        //}

        //Transpse A to be used with qpgen
        //for(w=0; w < m; w ++){
            //printf("A_t_trace: %lf || b: %lf\n",A_t_trace[w],b[w]);
            
        //}
        
        if(!matlessthaneq(A_t_trace,b,m,1)){
            results[i] = calc_depth(C_f,b,traces[i],m,n);
        }else{
            // Subtract A*x from b (b - A*x) 
            matsub(b_sub,m,1,A_t_trace,m,1);
            //Flip sign because qpgen works with Ax >= b
            matscaler(-1.0,b_sub,m,1);
            matscaler(-1.0,C_f_temp,m,n);
                
            //Transpse A to be used with qpgen
            transpose(C_f_temp,m,n , C_t);
            
            qpgen2_(G,a,&n,&n,sol,lagr,&results[i],C_t,b_sub,&n,&m,&meq,iact,&nact,iters,work,&ierr);
            results[i] = -1*sqrt(2*results[i]);
        }
        free(traces[i]);
    }
    //printf("ierr: %d\n" ,ierr);
    //printf("DONE\n");
}
