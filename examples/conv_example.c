#include "src/convolution.h"
#include <stdlib.h>
#include <stdio.h>
/* Here is a usage example */

#define DIMS 3

#define N0 20
#define N1 50
#define N2 40

int main()
{
    int i;
    // create a 3D matrix
    double *mat1 = (double*) malloc(sizeof(double) * N0 * N1 * N2); 
    int dim1[DIMS] = {N0, N1, N2};
    
    printf("\n\n mat1:\n");
    for(i = 0; i < (N0 * N1 * N2); ++i)
    {
        mat1[i] = rand() / (RAND_MAX + 1.);
        printf("%lf,", mat1[i]);        
    }
    // create another 3D matrix
    double *mat2 = (double*) malloc(sizeof(double) * N0 * N1 * N2); 
    int dim2[DIMS] = {N0, N1, N2};
    
    printf("\n\n mat2:\n");
    for(i = 0; i < (N0 * N1 * N2); ++i)
    {
        mat2[i] = rand() / (RAND_MAX + 1.);
        printf("%lf,", mat2[i]);
    }

    // create memory to put the result into
    double *res_mat = (double*) malloc(sizeof(double) * ( 2 * (N0 + N0 - 1) * (N1 + N1 - 1) * (N2 + N2 - 1) ) ); 
    int res_dim[DIMS];
    // now get the convolution result, and check for errors
    if( convnd(DIMS, mat1, dim1, mat2, dim2, res_mat, res_dim) != 0 )
    {
        printf("\n\n\n\n memory error encountered\n\n\n\n");
        return -1;
    }
    printf("\n\n res_mat:\n");    
    for(i = 0; i < (N0 * N1 * N2); ++i)
    {
        printf("%lf,", res_mat[i] );
    }
    return 0;
}
