/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
File:   main_2d.c
Author: koromodako
Date:   2016-11-24
Purpose:
    Contains MPI program for two dimensional matrix multiplication.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

int n, p;

int perfect_square(int n);

int main(int argc, char **argv) 
{
    /* variables */
    int my_n, my_rank, i, j, k, my_row, my_col, srp;
    double *a, *b, *c, *allC, start, sum, sumdiag, *rowA, *colB;
    MPI_Comm row_comm, col_comm;
    /* intput arguments parsing */
    if (argc<2) {
        printf("error: missing parameter 'n'\n");
        printf("usage: mpirun -np 4 --hostfile hostfile matmultmpi_2d <n>\n");
        exit(-1);
    }
    n=atoi(argv[1]);
    /* initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* check variables */
    if ((srp=perfect_square(p))==-1) { /* is p a perfect square ? */
        printf("error: p is not a perfect square (p=%d)\n", p);
        exit(-1);
    }
    if (n%srp!=0) { /* does srp divides n ? */
        printf("error: srp does not divide n (srp=%d, n=%d)\n", srp, n);
        exit(-1);
    }
    my_n=n/srp;
    my_row=my_rank/srp;
    my_col=my_rank-my_row*srp;
    /* initialize local matrices */
    a=calloc(my_n*my_n, sizeof(double));
    b=calloc(my_n*my_n, sizeof(double));
    c=calloc(my_n*my_n, sizeof(double));
    rowA=calloc(my_n*n, sizeof(double));
    colB=calloc(my_n*n, sizeof(double));
    for(i=0; i<my_n*my_n; i++) {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    /* wait for all processes to initialize local matrices */
    MPI_Barrier(MPI_COMM_WORLD);
    /* create  communicators for each row and column */
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_rank, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_rank, &col_comm);
    /* start timer */
    if (my_rank==0) {
        start = MPI_Wtime();
    }
    /* retrieve A's row and B's column */
    MPI_Allgather(a, my_n*my_n, MPI_DOUBLE, rowA, my_n*my_n, MPI_DOUBLE,
        row_comm);
    MPI_Allgather(b, my_n*my_n, MPI_DOUBLE, colB, my_n*my_n, MPI_DOUBLE,
        col_comm);
    /* compute matrix multiplication */
    for (i=0; i<my_n; ++i) {
        for (j=0; j<my_n; ++j) {
            sum=0.0;
            for (k=0; k<n; ++k) {
                sum+=rowA[i*n+k]*colB[k*my_n+j];
            }
            c[i*my_n+j]=sum;
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
    free(rowA);
    free(colB);
    /* gather result matrix */
    MPI_Gather(c, my_n*my_n, MPI_DOUBLE, allC, my_n*my_n, MPI_DOUBLE, 0,
        MPI_COMM_WORLD);
    /* check result matrix */
    if (my_rank==0) {
        for (i=0, sumdiag=0.0; i<n; i++) {
            sumdiag += allC[i*n+i];
        }
        printf("info: the trace of the resulting matrix is %f\n", sumdiag);
        free(allC);
    }
    /* free communicators */
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    /* finalize MPI */
    MPI_Finalize();
    free(a);
    free(b);
    free(c);
    return 0;
}

int perfect_square(int n)
{
    float sr=sqrt(n);
    int fsr=(int)sr;
    if (sr-(float)fsr!=0.0) {
        return -1;
    }
    return fsr;
}