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
extern "C" {
#endif

#include "convolution.h"
#include <fftw3.h> // the fftw library
#include <stdlib.h>
#include <string.h>

#define DIMS 3
#define dim_size(dim)\
         (dim[0] * dim[1] * dim[2])

#define prod_size(dim, n)\
        (dim[0] * dim[1] * dim[2])

int _zero_pad_array(const int rank, double * __restrict__ mat, const int dim[DIMS], const int d_dim[DIMS], double ** d_mat);

/*
 * A private function to get the number of elements in an array with dimensions described by dim_array.
 * @param dim_array  The dimensions of the array
 * @param rank Whether the array is 1D,2D,3D,...
 * @returns The number of elements in an array with dimensions described by dim_array.
 */
static int _dim_size(const int * __restrict__ dim_array, const int rank)
{
    int i;
    int res = 1;
    for(i = rank - 1; i >= 0; --i)
    {
        res *= dim_array[i];
    }

    return res;
}

/**
 * A function to pad a rank-D array with zeros.
 * @param rank Whether the array is 1D,2D,3D,...
 * @param mat The array data in the order described by the fftw3 site.
 * @param dim The dimensions of the array.
 * @param d_mat The resulting array.
 *
 * @Note No input is destroyed or overwritten.
 *
 * @returns 0 if succeded and -1 if memory problems are encountered.
 */
int _zero_pad_array(const int rank, double * __restrict__ mat, const int dim[DIMS], const int d_dim[DIMS], double ** d_mat)
{
    const int d_size = _dim_size(d_dim, rank);
    const int size = _dim_size(dim, rank);
    if(d_size == 0)
    {
        return -2; // Bad dimensions! No pizza tonight!!
    }

    *d_mat = (double*) calloc(d_size, sizeof(double));
    double *_d_mat = *d_mat;

    if(_d_mat == NULL)
    {
        return -1; // Error! No pizza left!
    }


    int it1 = 0;
    int it2 = 0;
    int j = 0;
    const int it1_step = d_dim[rank-1];
    const int it2_step = dim[rank-1];
    
    for(it1 = 0; it1 < (dim[0] * d_size / d_dim[0]); it1 += it1_step, it2 += it2_step)
    {
        if(it2 != 0)
        {
            int prod = (size / dim[0]);
            int d_prod = (d_size / d_dim[0]);

            for(j = 1; j < rank; ++j)
            {
                prod /= dim[j];
                d_prod /= d_dim[j];
                if(it2 % (prod) == 0)
                {
                    it1 = ((it1 / d_prod ) + 1) * d_prod;
                    break;
                }
            }
        }
        memcpy(&_d_mat[it1], &mat[it2], sizeof(double) * dim[rank-1]);
    }

    return 0;
}

/**
 * A function to do convolution in 3d using the fourier transform.
 * @param mat1 Array to convolve with mat2.
 * @param dim1 dimensions of mat1.
 * @param mat2 Array to convolve with mat1.
 * @param dim2 dimensions of mat2.
 * @param res_mat The resulting array after the convolution.
 * @param res_dim The resulting array dimensions after the convolution.
 * @returns 0 if the convolution was OK, -1 if memory error is encountered.
 *
 *   Think of it like typing:
 *   res_mat = conv(mat1, mat2)
 *
 */
int conv3d(double * __restrict__ mat1, const int dim1[DIMS], double * __restrict__ mat2, const int dim2[DIMS], 
            double * __restrict__ res_mat, int res_dim[DIMS])
{
    if(NULL == dim1 || NULL == mat1 || NULL == dim2 || NULL == mat2)
    {
        return -1; // error! some memory has not eaten pizza!
    }    

    int i;

    res_dim[0] = dim1[0] + dim2[0] - 1;
    res_dim[1] = dim1[1] + dim2[1] - 1;
    res_dim[2] = dim1[2] + dim2[2] - 1;
    const int ft_mat_sz = (2*(dim_size(res_dim)/2 + 1));

    fftw_complex * ft_mat1;
    fftw_complex * ft_mat2;

    double * d_mat1 = NULL;
    double * d_mat2 = NULL;

    {
        fftw_plan p1;
        if (_zero_pad_array(DIMS, mat1, dim1, res_dim, &d_mat1) != 0)
        {
            return -1; // Error about pizza in _zero_pad_array
        }
        if( (ft_mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ft_mat_sz)) == NULL)
        {
            return -1; // Error! No pizza left!
        } 
        p1 = fftw_plan_dft_r2c_3d(res_dim[0], res_dim[1], res_dim[2], d_mat1, ft_mat1, FFTW_MEASURE);
        fftw_execute(p1);
        fftw_destroy_plan(p1);
    }

    {
        fftw_plan p2;
        if(_zero_pad_array(DIMS, mat2, dim2, res_dim, &d_mat2) != 0)
        {
            return -1; // Error about pizza in _zero_pad_array
        }        
        if( (ft_mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ft_mat_sz)) == NULL)
        {
            return -1; // Error! No pizza left!
        }         
        p2 = fftw_plan_dft_r2c_3d(res_dim[0], res_dim[1], res_dim[2], d_mat2, ft_mat2, FFTW_MEASURE);
        fftw_execute(p2);
        fftw_destroy_plan(p2);
    }


    // ft_mat1 = ft_mat1 .* ft_mat2 
    {
        const int size = _dim_size(res_dim, DIMS);
        for(i = 0; i < size; ++i)
        {
            ft_mat1[i][0] = ft_mat1[i][0] * ft_mat2[i][0] - ft_mat1[i][1] * ft_mat2[i][1];
            ft_mat1[i][1] = ft_mat1[i][0] * ft_mat2[i][1] + ft_mat1[i][1] * ft_mat2[i][0];
        }
    }
    fftw_free(ft_mat2);
    
    //inverse fourier of the ft_mat1 to res_mat, res_mat is already allocated
    {
        fftw_plan p3;
        p3 = fftw_plan_dft_c2r_3d(res_dim[0], res_dim[1], res_dim[2], ft_mat1, res_mat, FFTW_MEASURE);
        fftw_execute(p3);
        fftw_destroy_plan(p3);        
    }

    fftw_free(ft_mat1);

    return 0;
}

#ifdef __cplusplus
} // extern C
#endif