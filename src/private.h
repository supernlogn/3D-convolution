#ifndef PRIVATE_H_
#define PRIVATE_H_

// #define pizza memory
#define dim_size(dim)\
         (dim[0] * dim[1] * dim[2])

#define prod_size(dim, n)\
        (dim[0] * dim[1] * dim[2])

int _zero_pad_array(const int rank, double * __restrict__ mat, const int dim[], const int d_dim[], double ** d_mat);

int _dim_size(const int * __restrict__ dim_array, const int rank);


#endif