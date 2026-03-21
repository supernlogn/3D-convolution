/**
 * This is a plug and play library for performing 3D convolution in C/C++.
 * MIT License
 * Copyright (C) 2017 supernlogn(ath.ioannis94@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifdef __cplusplus
extern "C"
{
#endif

#include "convolution.h"
#include "private.h"
#include <fftw3.h>  // the fftw library
#include <stdlib.h>
#include <string.h>

struct convnd_plan
{
    int rank;
    int size;
    int ft_complex_sz;
    int *dim1;
    int *dim2;
    int *res_dim;
    double *d_tensor1;
    double *d_tensor2;
    double *res_tensor;
    fftw_complex *ft_tensor1;
    fftw_complex *ft_tensor2;
    fftw_plan p1;
    fftw_plan p2;
    fftw_plan p3;
};


/*
    * A private function to get the number of elements in an array with dimensions described by
    * dim_array.
    * @param dim_array  The dimensions of the array
    * @param rank Whether the array is 1D,2D,3D,...
    * @returns The number of elements in an array with dimensions described by dim_array.
    */
int _dim_size(const int *__restrict__ dim_array, const int rank)
{
    int i;
    int res = 1;
    for (i = 0; i < rank; ++i)
    {
        res *= dim_array[i];
    }

    return res;
}

static int _copy_into_padded(const int rank, const double *__restrict__ src,
                                const int *__restrict__ src_dim,
                                const int *__restrict__ padded_dim, double *__restrict__ padded)
{
    const int src_size = _dim_size(src_dim, rank);
    const int padded_size = _dim_size(padded_dim, rank);

    if (src_size <= 0 || padded_size <= 0)
    {
        return -1;
    }

    memset(padded, 0, (size_t) padded_size * sizeof(double));

    for (int src_index = 0; src_index < src_size; ++src_index)
    {
        int coords[rank];
        int rem = src_index;
        int padded_index = 0;

        for (int i = rank - 1; i >= 0; --i)
        {
            coords[i] = rem % src_dim[i];
            rem /= src_dim[i];
        }

        for (int i = 0; i < rank; ++i)
        {
            padded_index = padded_index * padded_dim[i] + coords[i];
        }

        padded[padded_index] = src[src_index];
    }

    return 0;
}

convnd_plan *convnd_plan_create(const int rank, const int *__restrict__ dim1,
                                const int *__restrict__ dim2, int *__restrict__ res_dim)
{
    if (rank <= 0 || dim1 == NULL || dim2 == NULL)
    {
        return NULL;
    }

    convnd_plan *plan = (convnd_plan *) calloc(1, sizeof(convnd_plan));
    if (plan == NULL)
    {
        return NULL;
    }

    plan->rank = rank;
    plan->dim1 = (int *) malloc((size_t) rank * sizeof(int));
    plan->dim2 = (int *) malloc((size_t) rank * sizeof(int));
    plan->res_dim = (int *) malloc((size_t) rank * sizeof(int));

    if (plan->dim1 == NULL || plan->dim2 == NULL || plan->res_dim == NULL)
    {
        convnd_plan_destroy(plan);
        return NULL;
    }

    for (int i = 0; i < rank; ++i)
    {
        if (dim1[i] <= 0 || dim2[i] <= 0)
        {
            convnd_plan_destroy(plan);
            return NULL;
        }
        plan->dim1[i] = dim1[i];
        plan->dim2[i] = dim2[i];
        plan->res_dim[i] = dim1[i] + dim2[i] - 1;
        if (res_dim != NULL)
        {
            res_dim[i] = plan->res_dim[i];
        }
    }

    plan->size = _dim_size(plan->res_dim, rank);
    plan->ft_complex_sz =
        (plan->size / plan->res_dim[rank - 1]) * (plan->res_dim[rank - 1] / 2 + 1);

    if (plan->size <= 0 || plan->ft_complex_sz <= 0)
    {
        convnd_plan_destroy(plan);
        return NULL;
    }

    plan->d_tensor1 = (double *) calloc((size_t) plan->size, sizeof(double));
    plan->d_tensor2 = (double *) calloc((size_t) plan->size, sizeof(double));
    plan->res_tensor = (double *) calloc((size_t) plan->size, sizeof(double));
    plan->ft_tensor1 =
        (fftw_complex *) fftw_malloc((size_t) plan->ft_complex_sz * sizeof(fftw_complex));
    plan->ft_tensor2 =
        (fftw_complex *) fftw_malloc((size_t) plan->ft_complex_sz * sizeof(fftw_complex));

    if (plan->d_tensor1 == NULL || plan->d_tensor2 == NULL || plan->res_tensor == NULL
        || plan->ft_tensor1 == NULL || plan->ft_tensor2 == NULL)
    {
        convnd_plan_destroy(plan);
        return NULL;
    }

    plan->p1 = fftw_plan_dft_r2c(rank, plan->res_dim, plan->d_tensor1, plan->ft_tensor1,
                                    FFTW_ESTIMATE);
    plan->p2 = fftw_plan_dft_r2c(rank, plan->res_dim, plan->d_tensor2, plan->ft_tensor2,
                                    FFTW_ESTIMATE);
    plan->p3 = fftw_plan_dft_c2r(rank, plan->res_dim, plan->ft_tensor1, plan->res_tensor,
                                    FFTW_ESTIMATE);

    if (plan->p1 == NULL || plan->p2 == NULL || plan->p3 == NULL)
    {
        convnd_plan_destroy(plan);
        return NULL;
    }

    return plan;
}

int convnd_plan_execute(convnd_plan *plan, double *__restrict__ tensor1,
                        double *__restrict__ tensor2, double *__restrict__ res_tensor)
{
    if (plan == NULL || tensor1 == NULL || tensor2 == NULL || res_tensor == NULL)
    {
        return -1;
    }

    if (_copy_into_padded(plan->rank, tensor1, plan->dim1, plan->res_dim, plan->d_tensor1) != 0)
    {
        return -1;
    }

    if (_copy_into_padded(plan->rank, tensor2, plan->dim2, plan->res_dim, plan->d_tensor2) != 0)
    {
        return -1;
    }

    fftw_execute(plan->p1);
    fftw_execute(plan->p2);

    for (int i = 0; i < plan->ft_complex_sz; ++i)
    {
        const double real1 = plan->ft_tensor1[i][0];
        const double imag1 = plan->ft_tensor1[i][1];
        const double real2 = plan->ft_tensor2[i][0];
        const double imag2 = plan->ft_tensor2[i][1];

        plan->ft_tensor1[i][0] = real1 * real2 - imag1 * imag2;
        plan->ft_tensor1[i][1] = real1 * imag2 + imag1 * real2;
    }

    fftw_execute(plan->p3);

    const double df_size = (double) plan->size;
    for (int i = 0; i < plan->size; ++i)
    {
        res_tensor[i] = plan->res_tensor[i] / df_size;
    }

    return 0;
}

int convnd_plan_get_result_dim(const convnd_plan *plan, int *__restrict__ res_dim)
{
    if (plan == NULL || res_dim == NULL)
    {
        return -1;
    }

    for (int i = 0; i < plan->rank; ++i)
    {
        res_dim[i] = plan->res_dim[i];
    }

    return 0;
}

int convnd_plan_result_size(const convnd_plan *plan)
{
    if (plan == NULL)
    {
        return -1;
    }

    return plan->size;
}

void convnd_plan_destroy(convnd_plan *plan)
{
    if (plan == NULL)
    {
        return;
    }

    if (plan->p1 != NULL)
    {
        fftw_destroy_plan(plan->p1);
    }
    if (plan->p2 != NULL)
    {
        fftw_destroy_plan(plan->p2);
    }
    if (plan->p3 != NULL)
    {
        fftw_destroy_plan(plan->p3);
    }

    if (plan->ft_tensor1 != NULL)
    {
        fftw_free(plan->ft_tensor1);
    }
    if (plan->ft_tensor2 != NULL)
    {
        fftw_free(plan->ft_tensor2);
    }

    free(plan->d_tensor1);
    free(plan->d_tensor2);
    free(plan->res_tensor);
    free(plan->dim1);
    free(plan->dim2);
    free(plan->res_dim);
    free(plan);
}

/**
 * A function to pad a rank-D array with zeros.
 * @param rank Whether the array is 1D,2D,3D,...
 * @param mat The array data in the order described by the fftw3 site.
 * @param dim The dimensions of the array.
 * @param d_dim The resulting tensor dimensions after padding
 * @param d_mat The resulting array.
 *
 * @Note No input is destroyed or overwritten.
 *
 * @returns 0 if succeeded and -1 if memory problems are encountered.
 */
int _zero_pad_array(const int rank, double *__restrict__ mat, const int dim[],
                    const int d_dim[], double **d_mat)
{
    const int d_size = _dim_size(d_dim, rank);
    const int size = _dim_size(dim, rank);
    if (d_size == 0)
    {
        return -2;  // Bad dimensions! No pizza tonight!!
    }

    *d_mat = (double *) calloc(d_size, sizeof(double));
    double *_d_mat = *d_mat;

    if (_d_mat == NULL)
    {
        return -1;  // Error! No pizza left!
    }

    int *iterators = (int *) calloc(rank, sizeof(int));
    int total_num_els = _dim_size(d_dim, rank);
    int in_zero_area = 0;
    int numels_orig = 0;
    for (int numels = 0; numels < total_num_els; ++numels)
    {
        in_zero_area = 0;
        for (int i = 0; i < rank; ++i)
        {
            in_zero_area |= (iterators[i] >= dim[i]);
        }
        if (0 == in_zero_area)
        {
            (*d_mat)[numels] = mat[numels_orig];
            ++numels_orig;
        }
        for (int i = rank - 1; i >= 0; --i)
        {
            iterators[i] += 1;
            if (iterators[i] < d_dim[i])
            {
                break;
            }
            else
            {
                iterators[i] = 0;
            }
        }
    }
    free(iterators);


    return 0;
}

/**
 * A function to do convolution in n dimensional tensors using the fourier transform.
 * @param tensor1 Array to convolve with tensor2.
 * @param dim1 dimensions of tensor1.
 * @param tensor2 Array to convolve with tensor1.
 * @param dim2 dimensions of tensor2.
 * @param res_tensor The resulting tensor after the convolution.
 * @param res_dim The resulting tensor dimensions after the convolution.
 * @returns 0 if the convolution was OK, -1 if memory error is encountered.
 *
 *   Think of it like typing:
 *   res_tensor = conv(tensor1, tensor2)
 *
 */
int convnd(const int rank, double *__restrict__ tensor1, const int *__restrict__ dim1,
            double *__restrict__ tensor2, const int *__restrict__ dim2,
            double *__restrict__ res_tensor, int *__restrict__ res_dim)
{
    if (NULL == dim1 || NULL == tensor1 || NULL == dim2 || NULL == tensor2 || NULL == res_dim)
    {
        return -1;  // error! some memory has not eaten pizza!
    }

    int i;
    for (i = rank - 1; i >= 0; --i)
    {
        res_dim[i] = dim1[i] + dim2[i] - 1;
    }

    const int ft_complex_sz =
        (_dim_size(res_dim, rank) / res_dim[rank - 1]) * (res_dim[rank - 1] / 2 + 1);

    fftw_complex *ft_tensor1;
    fftw_complex *ft_tensor2;

    double *d_tensor1 = NULL;
    double *d_tensor2 = NULL;

    // ft_tensor1 = fft(tensor1)
    {
        fftw_plan p1;
        if (_zero_pad_array(rank, tensor1, dim1, res_dim, &d_tensor1) != 0)
        {
            return -1;  // Error about pizza in _zero_pad_array
        }
        if ((ft_tensor1 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * ft_complex_sz))
            == NULL)
        {
            return -1;  // Error! No pizza left!
        }
        p1 = fftw_plan_dft_r2c(rank, res_dim, d_tensor1, ft_tensor1, FFTW_ESTIMATE);
        fftw_execute(p1);
        fftw_destroy_plan(p1);
    }

    // ft_tensor2 = fft(tensor2)
    {
        fftw_plan p2;
        if (_zero_pad_array(rank, tensor2, dim2, res_dim, &d_tensor2) != 0)
        {
            return -1;  // Error about pizza in _zero_pad_array
        }
        if ((ft_tensor2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * ft_complex_sz))
            == NULL)
        {
            return -1;  // Error! No pizza left!
        }
        p2 = fftw_plan_dft_r2c(rank, res_dim, d_tensor2, ft_tensor2, FFTW_ESTIMATE);
        fftw_execute(p2);
        fftw_destroy_plan(p2);
    }

    const int size = _dim_size(res_dim, rank);

    // ft_tensor1 = ft_tensor1 .* ft_tensor2
    {
        for (i = ft_complex_sz - 1; i >= 0; --i)
        {
            const double real1 = ft_tensor1[i][0];
            const double imag1 = ft_tensor1[i][1];
            const double real2 = ft_tensor2[i][0];
            const double imag2 = ft_tensor2[i][1];

            ft_tensor1[i][0] = real1 * real2 - imag1 * imag2;
            ft_tensor1[i][1] = real1 * imag2 + imag1 * real2;
        }
    }
    fftw_free(ft_tensor2);
    free(d_tensor1);
    free(d_tensor2);

    // inverse fourier of the ft_tensor1 to res_tensor, res_tensor is already allocated
    {
        fftw_plan p3;
        p3 = fftw_plan_dft_c2r(rank, res_dim, ft_tensor1, res_tensor,
                                FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
        fftw_execute(p3);
        fftw_destroy_plan(p3);
    }
    fftw_free(ft_tensor1);

    // normalize the resulting matrix
    {
        const double df_size = (double) size;
        for (i = size - 1; i >= 0; --i)
        {
            res_tensor[i] /= df_size;
        }
    }


    return 0;
}

#ifdef __cplusplus
}  // extern C
#endif
