#include <stdlib.h>
#include <stdio.h>
#include "private.h"

// Test the zero padding
void fill_mat(double * mat, const int length, const double val) {
    for (int i = 0; i < length; ++i) {
        mat[i] = val;
    }
}

void print_mat(const int rank, const double *mat, const int dims[])
{
    int * iterators = (int*) calloc(rank, sizeof(int));
    int total_num_els = _dim_size(dims, rank);
    for (int numels = 0; numels < total_num_els; numels += dims[rank-1])
    {
        const int lli = dims[rank-1];
        for (int i = 0; i < lli; ++i) // iterators[rank-1]
        {
            printf("%0.3f ",mat[i]);
            
        }
        printf("\n");
        int keeper = 1;
        for(int i = rank-2; i >= 0; --i)
        {
            iterators[i] += keeper;
            if (iterators[i] < dims[i])
            {
              keeper = 0;
            } else {
              printf("\n");
              keeper = 1;
              iterators[i] = 0;
            }
        }
    }
    free(iterators);
}

void test1(const int add_to_N0, const int add_to_N1)
{
  int rank = 2;
  int DIMS = 2;
  int N0 = 5;
  int N1 = 8;
  double *mat = (double *)malloc(sizeof(double) * N0 * N1);
  fill_mat(mat, N0 * N1, 1.0);
  int dim[] = {N0, N1};
  int res_dim[] = {N0 + add_to_N0, N1 + add_to_N1};
  double * final_mat;
  _zero_pad_array(rank, mat, dim, res_dim, &final_mat);
  printf("dims = {%d,%d}\n", res_dim[0], res_dim[1]);
  print_mat(rank, final_mat, res_dim);
}

void main(int argc, char *argv[])
{
  test1(4,3);

  return;
}