#include "src/convolution.h"
#include <stdlib.h>
#include <stdio.h>
/* Here is a usage example */

#define N0 10 
#define N1 50
#define N2 50

int main()
{
    int i;
    // create a 3D matrix
    double *mat1 = (double*) malloc(sizeof(double) * N0 * N1 * N2); 
    int dim1[3] = {N0, N1, N2};
    

    for(i = 0; i < (N0 * N1 * N2); ++i)
    {
        mat1[i] = rand() / (RAND_MAX + 1.);;
    }
    // create another 3D matrix
    double *mat2 = (double*) malloc(sizeof(double) * N0 * N1 * N2); 
    int dim2[3] = {N0, N1, N2};
    
    for(i = 0; i < (N0 * N1 * N2); ++i)
    {
        mat2[i] = rand() / (RAND_MAX + 1.);
    }
    // create memory to put the result into
    double *res_mat = (double*) malloc(sizeof(double) * ( 2 * (N0 + N0 - 1) * (N1 + N1 - 1) * (N2 + N2 - 1) ) ); 
    int res_dim[3];
    // now get the convolution result, and check for errors
    if( conv3d(mat1, dim1, mat2, dim2, res_mat, res_dim) != 0 )
    {
        printf("\n\n\n\n memory error encountered\n\n\n\n");
        return -1;
    }

    return 0;
}
