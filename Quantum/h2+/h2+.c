/**
 *  Quantum Simulation of a "Transistor"
 *   by Aidan Macdonald
 *
 */

#include <stdlib.h>

/** Take parts of the wave function and
 * evolve them according to the hamiltonian for a certain number of steps
 * For proper time stepping, this method appears to conserve probability well
 */

void evolve(double *__restrict__ psi_r, double *__restrict__ psi_i,
	    double * __restrict__ qp1, double * __restrict__ qp2,
	    int i_max, int j_max, double x0, double y0, double dx, double dt,
	    double Mp, double Me, double k) {

  double * __restrict__ psi_r_tmp, * __restrict__ psi_i_tmp, dP1, dP2;
  psi_r_tmp = malloc(sizeof(double)*i_max*j_max);
  psi_i_tmp = malloc(sizeof(double)*i_max*j_max);

  int t, i, j;

  for (i=0;i<i_max*j_max;i++)
    psi_r_tmp[i] = psi_i_tmp[i] = 0.0;
  
  for (t=0;t<steps;t++) {
    // First Partial Step
#pragma omp parallel
    {
#pragma omp for
      for (i=1;i<i_max-1;i++) {
	for (j=1;j<j_max-1;j++) {
	  
	}
      }
      qp1[0] += dt*qp1[1]/Mp;
      qp2[0] += dt*qp2[1]/Mp;
      qp1[1] += 

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

