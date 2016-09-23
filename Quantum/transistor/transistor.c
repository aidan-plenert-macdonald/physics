/**
 *  Quantum Simulation of a "Transistor"
 *   by Aidan Macdonald
 *
 */

#include <stdlib.h>
#include <stdio.h>

void evolve_row(const double * __restrict__ psi_r, double * __restrict__ psi_r_tmp,
		const double * __restrict__ psi_i, double * __restrict__ psi_i_tmp,
		const int w0, const int w1, const int h0, const int h1, 
		const int j_max, const int i, const double c0, const double c1) {
  double ctmp;
  int j;
  for (j=1;j<j_max-1;j++) {
    ctmp = 1.0 + (w0 < i && i < w1 && (j < h0 || h1 < j) ? c1 : 0.0);
    psi_r_tmp[i*j_max + j] = (ctmp*psi_r[i*j_max + j] + 
			      c0*(psi_i[i*j_max + j+1] - 4*psi_i[i*j_max + j] + 
				  psi_i[i*j_max + j-1] +
				  psi_i[(i+1)*j_max + j] + psi_i[(i-1)*j_max + j]));
    psi_i_tmp[i*j_max + j] = (ctmp*psi_i[i*j_max + j] - 
			      c0*(psi_r[i*j_max + j+1] - 4*psi_r[i*j_max + j] + 
				  psi_r[i*j_max + j-1] +
				  psi_r[(i+1)*j_max + j] + psi_r[(i-1)*j_max + j]));
  }  
}

/** Take parts of the wave function and
 * evolve them according to the hamiltonian for a certain number of steps
 * For proper time stepping, this method appears to conserve probability well
 */

void evolve(double *__restrict__ psi_r, double *__restrict__ psi_i,
	    int i_max, int j_max, double x0, double y0,
	    double dx, double dt, double width, double height,
	    double V0, double m, int steps) {

  double * __restrict__ psi_r_tmp, * __restrict__ psi_i_tmp;
  psi_r_tmp = malloc(sizeof(double)*i_max*j_max);
  psi_i_tmp = malloc(sizeof(double)*i_max*j_max);

  const double c0 = -0.5*dt/(2*m*dx*dx), c1 = 0.5*dt*V0;
  const int w0 = (-x0 - width)/dx, w1 = (-x0 + width)/dx;
  const int h0 = (-y0 - height)/dx, h1 = (-y0 + height)/dx;
  int t, i, j;

  for (i=0;i<i_max*j_max;i++)
    psi_r_tmp[i] = psi_i_tmp[i] = 0.0;
  
  for (t=0;t<steps;t++) {
    // First Partial Step
#pragma omp parallel
    {
#pragma omp for
      for (i=1;i<i_max-1;i++) {
	evolve_row(psi_r, psi_r_tmp, psi_i, psi_i_tmp, w0, w1, h0, h1, j_max, i, c0, c1);
      }

#pragma omp barrier
      // Second Partial Step
#pragma omp for
      for (i=1;i<i_max-1;i++) {
	evolve_row(psi_r_tmp, psi_r, psi_i_tmp, psi_i, w0, w1, h0, h1, j_max, i, c0, c1);
      }
    }
  } // END TIME STEPPING

  free(psi_r_tmp);
  free(psi_i_tmp);
}

