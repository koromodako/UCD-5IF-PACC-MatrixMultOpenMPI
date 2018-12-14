/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
File:   main_1d.c
Author: koromodako
Date:   2016-11-23
Purpose:
    Contains MPI program for one dimensional matrix multiplication.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>

int n, p;

int main(int argc, char **argv) 
{
    /* variables */
    int my_n, my_rank, i, j, k;
    double *a, *b, *c, *allC, start, sum, sumdiag, *allB;
    /* intput arguments parsing */
    if (argc<2) {
        printf("error: missing parameter 'n'\n");
        printf("usage: mpirun -np 4 --hostfile hostfile matmultmpi_1d <n>\n");
        exit(-1);
    }
    n=atoi(argv[1]);
    /* initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* check variables */
    if (n%p!=0) { /* does p divides n ? */
        printf("error: p does not divide n (p=%d, n=%d)\n", p, n);
        exit(-1);
    }
    my_n=n/p;
    /* initialize local matrices */
    a=calloc(my_n*n, sizeof(double));
    b=calloc(my_n*n, sizeof(double));
    c=calloc(my_n*n, sizeof(double));
    allB=calloc(n*n, sizeof(double));
    for(i=0; i<my_n*n; i++) {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    /* wait for all processes to initialize local matrices */
    MPI_Barrier(MPI_COMM_WORLD);
    /* start timer */
    if (my_rank==0) {
        start = MPI_Wtime();
    }
    /* retrieve all B parts */
    MPI_Allgather(b, my_n*n, MPI_DOUBLE, allB, my_n*n, MPI_DOUBLE,
        MPI_COMM_WORLD);
    /* compute matrix multiplication */
    for (i=0; i<my_n; i++) {
        for (j=0; j<n; j++) {
            sum=0.0;
            for (k=0; k<n; k++) {
                sum += a[i*n+k]*allB[k*n+j];
            }
            c[i*n+j] = sum;
        }
    }
    /* wait for all processes to compute local submatrix */
    MPI_Barrier(MPI_COMM_WORLD);
    /* stop timer and alocate result matrix */
    if (my_rank==0) {
        printf("It took %f seconds to multiply 2 %dx%d matrices.\n",
            MPI_Wtime()-start, n, n);
        allC=malloc(n*n*sizeof(double));    
    }
    /* free local memory allocated used to store external parts */
    free(allB);
    /* gather result matrix */
    MPI_Gather(c, my_n*n, MPI_DOUBLE, allC, my_n*n, MPI_DOUBLE, 0,
        MPI_COMM_WORLD);
    /* check result matrix */
    if (my_rank==0) {
        for (i=0, sumdiag=0.0; i<n; i++) {
            sumdiag += allC[i*n+i];
        }
        printf("info: the trace of the resulting matrix is %f\n", sumdiag);
        free(allC);
    }
    /* finalize MPI */
    MPI_Finalize();
    free(a);
    free(b);
    free(c);
    return 0;
}