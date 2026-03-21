#include "src/convolution.h"

#include <stdio.h>
#include <stdlib.h>

#define DIMS 3

#define N0 6
#define N1 5
#define N2 4

static int tensor_size(const int dims[], const int rank)
{
    int size = 1;

    for (int i = 0; i < rank; ++i)
    {
        size *= dims[i];
    }

    return size;
}

static void fill_random(double *values, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        values[i] = rand() / (RAND_MAX + 1.0);
    }
}

static void print_preview(const char *label, const double *values, const int length)
{
    const int preview = length < 8 ? length : 8;

    printf("%s (%d values): ", label, length);
    for (int i = 0; i < preview; ++i)
    {
        printf("%.5f", values[i]);
        if (i + 1 < preview)
        {
            printf(", ");
        }
    }
    if (length > preview)
    {
        printf(", ...");
    }
    printf("\n");
}

int main(void)
{
    int dim1[DIMS] = {N0, N1, N2};
    int dim2[DIMS] = {N0, N1, N2};
    int res_dim[DIMS] = {0};

    const int mat1_size = tensor_size(dim1, DIMS);
    const int mat2_size = tensor_size(dim2, DIMS);
    const int res_size = (dim1[0] + dim2[0] - 1) *
                         (dim1[1] + dim2[1] - 1) *
                         (dim1[2] + dim2[2] - 1);

    double *mat1 = (double *)malloc((size_t)mat1_size * sizeof(double));
    double *mat2 = (double *)malloc((size_t)mat2_size * sizeof(double));
    double *res_mat = (double *)malloc((size_t)res_size * sizeof(double));

    if (mat1 == NULL || mat2 == NULL || res_mat == NULL)
    {
        printf("Allocation failed.\n");
        free(mat1);
        free(mat2);
        free(res_mat);
        return 1;
    }

    srand(7);
    fill_random(mat1, mat1_size);
    fill_random(mat2, mat2_size);

    if (convnd(DIMS, mat1, dim1, mat2, dim2, res_mat, res_dim) != 0)
    {
        printf("Convolution failed.\n");
        free(mat1);
        free(mat2);
        free(res_mat);
        return 1;
    }

    printf("Input A dims: [%d, %d, %d]\n", dim1[0], dim1[1], dim1[2]);
    printf("Input B dims: [%d, %d, %d]\n", dim2[0], dim2[1], dim2[2]);
    printf("Result dims:  [%d, %d, %d]\n", res_dim[0], res_dim[1], res_dim[2]);

    print_preview("A preview", mat1, mat1_size);
    print_preview("B preview", mat2, mat2_size);
    print_preview("Result preview", res_mat, res_size);

    free(mat1);
    free(mat2);
    free(res_mat);
    return 0;
}
