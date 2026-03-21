#ifndef TESTS_TEST_CONVOLUTION_H
#define TESTS_TEST_CONVOLUTION_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution.h"

static int test_dim_size(const int dims[], const int rank)
{
    int size = 1;

    for (int i = 0; i < rank; ++i)
    {
        size *= dims[i];
    }

    return size;
}

static void test_flat_to_coords(int flat_index, const int dims[], const int rank, int coords[])
{
    for (int i = rank - 1; i >= 0; --i)
    {
        coords[i] = flat_index % dims[i];
        flat_index /= dims[i];
    }
}

static int test_coords_to_flat(const int coords[], const int dims[], const int rank)
{
    int flat_index = 0;

    for (int i = 0; i < rank; ++i)
    {
        flat_index = flat_index * dims[i] + coords[i];
    }

    return flat_index;
}

static void naive_convnd(const int rank, const double tensor1[], const int dim1[],
                         const double tensor2[], const int dim2[], double result[],
                         int result_dims[])
{
    const int size1 = test_dim_size(dim1, rank);
    const int result_size = test_dim_size(result_dims, rank);
    int *result_coords = (int *) calloc(rank, sizeof(int));
    int *tensor1_coords = (int *) calloc(rank, sizeof(int));
    int *tensor2_coords = (int *) calloc(rank, sizeof(int));

    for (int i = 0; i < rank; ++i)
    {
        result_dims[i] = dim1[i] + dim2[i] - 1;
    }

    for (int result_index = 0; result_index < result_size; ++result_index)
    {
        double sum = 0.0;

        test_flat_to_coords(result_index, result_dims, rank, result_coords);
        for (int tensor1_index = 0; tensor1_index < size1; ++tensor1_index)
        {
            int inside = 1;

            test_flat_to_coords(tensor1_index, dim1, rank, tensor1_coords);
            for (int i = 0; i < rank; ++i)
            {
                tensor2_coords[i] = result_coords[i] - tensor1_coords[i];
                inside &= tensor2_coords[i] >= 0 && tensor2_coords[i] < dim2[i];
            }

            if (inside)
            {
                const int tensor2_index = test_coords_to_flat(tensor2_coords, dim2, rank);
                sum += tensor1[tensor1_index] * tensor2[tensor2_index];
            }
        }

        result[result_index] = sum;
    }

    free(result_coords);
    free(tensor1_coords);
    free(tensor2_coords);
}

static int compare_double_arrays(const double lhs[], const double rhs[], const int length,
                                 const double tolerance)
{
    for (int i = 0; i < length; ++i)
    {
        if (fabs(lhs[i] - rhs[i]) > tolerance)
        {
            return i + 1;
        }
    }

    return 0;
}

static void fill_deterministic_values(double values[], const int length, unsigned int seed)
{
    unsigned int state = seed;

    for (int i = 0; i < length; ++i)
    {
        state = state * 1664525u + 1013904223u;
        values[i] = ((double) (state % 2001u) - 1000.0) / 137.0;
    }
}

static int run_convnd_case(const char *name, const int rank, const int dim1[],
                           const double tensor1[], const int dim2[], const double tensor2[])
{
    const int size1 = test_dim_size(dim1, rank);
    const int size2 = test_dim_size(dim2, rank);
    int expected_dims[8] = {0};
    int actual_dims[8] = {0};
    double *tensor1_copy = (double *) malloc((size_t) size1 * sizeof(double));
    double *tensor2_copy = (double *) malloc((size_t) size2 * sizeof(double));
    double *expected = NULL;
    double *actual = NULL;
    int result_size = 1;
    int failure_index = 0;
    int status = 0;

    memcpy(tensor1_copy, tensor1, (size_t) size1 * sizeof(double));
    memcpy(tensor2_copy, tensor2, (size_t) size2 * sizeof(double));

    for (int i = 0; i < rank; ++i)
    {
        expected_dims[i] = dim1[i] + dim2[i] - 1;
        result_size *= expected_dims[i];
    }

    expected = (double *) calloc((size_t) result_size, sizeof(double));
    actual = (double *) calloc((size_t) result_size, sizeof(double));

    naive_convnd(rank, tensor1, dim1, tensor2, dim2, expected, expected_dims);

    status = convnd(rank, tensor1_copy, dim1, tensor2_copy, dim2, actual, actual_dims);
    if (status != 0)
    {
        printf("[FAIL] %s returned %d\n", name, status);
        free(tensor1_copy);
        free(tensor2_copy);
        free(expected);
        free(actual);
        return 1;
    }

    for (int i = 0; i < rank; ++i)
    {
        if (actual_dims[i] != expected_dims[i])
        {
            printf("[FAIL] %s produced wrong dimension at axis %d: expected %d got %d\n", name, i,
                   expected_dims[i], actual_dims[i]);
            free(tensor1_copy);
            free(tensor2_copy);
            free(expected);
            free(actual);
            return 1;
        }
    }

    failure_index = compare_double_arrays(actual, expected, result_size, 1e-8);
    if (failure_index != 0)
    {
        printf("[FAIL] %s mismatch at output index %d: expected %.10f got %.10f\n", name,
               failure_index - 1, expected[failure_index - 1], actual[failure_index - 1]);
        free(tensor1_copy);
        free(tensor2_copy);
        free(expected);
        free(actual);
        return 1;
    }

    printf("[PASS] %s\n", name);

    free(tensor1_copy);
    free(tensor2_copy);
    free(expected);
    free(actual);
    return 0;
}

static int test_convnd_rank3_scalar_kernel(void)
{
    const int rank = 3;
    const int dim1[] = {2, 2, 2};
    const int dim2[] = {1, 1, 1};
    const double tensor1[] = {1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 1.5};
    const double tensor2[] = {3.0};

    return run_convnd_case("rank3 scalar kernel", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank3_rectangular(void)
{
    const int rank = 3;
    const int dim1[] = {2, 1, 3};
    const int dim2[] = {1, 2, 2};
    const double tensor1[] = {1.0, -1.0, 2.0, 0.5, 3.0, -2.0};
    const double tensor2[] = {2.0, -1.0, 0.0, 4.0};

    return run_convnd_case("rank3 rectangular tensors", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank4_mixed_values(void)
{
    const int rank = 4;
    const int dim1[] = {2, 2, 1, 2};
    const int dim2[] = {1, 2, 2, 1};
    const double tensor1[] = {1.0, 0.0, -2.0, 3.0, 4.0, -1.0, 2.0, 1.5};
    const double tensor2[] = {2.0, -1.0, 0.5, 3.0};

    return run_convnd_case("rank4 mixed values", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank5_sparse_pattern(void)
{
    const int rank = 5;
    const int dim1[] = {2, 1, 2, 1, 2};
    const int dim2[] = {1, 2, 1, 2, 1};
    const double tensor1[] = {1.0, 0.0, 0.0, -1.0, 2.5, 0.0, 0.0, 3.0};
    const double tensor2[] = {1.0, -2.0, 0.5, 4.0};

    return run_convnd_case("rank5 sparse pattern", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank6_high_rank(void)
{
    const int rank = 6;
    const int dim1[] = {2, 1, 2, 1, 2, 1};
    const int dim2[] = {1, 2, 1, 2, 1, 2};
    const double tensor1[] = {1.0, -1.0, 2.0, -2.0, 0.5, 1.5, -0.5, 3.0};
    const double tensor2[] = {2.0, 0.0, -1.0, 1.0, 3.0, -2.0, 0.5, 4.0};

    return run_convnd_case("rank6 high rank", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank3_zero_tensors(void)
{
    const int rank = 3;
    const int dim1[] = {2, 2, 3};
    const int dim2[] = {2, 1, 2};
    const double tensor1[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double tensor2[] = {0.0, 0.0, 0.0, 0.0};

    return run_convnd_case("rank3 zero tensors", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank4_identity_kernel(void)
{
    const int rank = 4;
    const int dim1[] = {2, 2, 2, 2};
    const int dim2[] = {1, 1, 1, 1};
    const double tensor1[] = {1.0, -2.0, 3.5, 4.0, -1.0, 0.0, 2.25, -3.0,
                              5.0, -4.5, 1.5, 2.0, -2.5, 3.0, 0.75, -1.25};
    const double tensor2[] = {1.0};

    return run_convnd_case("rank4 identity kernel", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank5_identity_kernel(void)
{
    const int rank = 5;
    const int dim1[] = {2, 2, 1, 2, 2};
    const int dim2[] = {1, 1, 1, 1, 1};
    const double tensor1[] = {1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0,
                              0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5};
    const double tensor2[] = {1.0};

    return run_convnd_case("rank5 identity kernel", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank3_large_deterministic(void)
{
    const int rank = 3;
    const int dim1[] = {3, 3, 4};
    const int dim2[] = {2, 2, 3};
    double tensor1[36];
    double tensor2[12];

    fill_deterministic_values(tensor1, 36, 17u);
    fill_deterministic_values(tensor2, 12, 91u);

    return run_convnd_case("rank3 larger deterministic", rank, dim1, tensor1, dim2, tensor2);
}

static int test_convnd_rank4_large_deterministic(void)
{
    const int rank = 4;
    const int dim1[] = {2, 3, 2, 3};
    const int dim2[] = {2, 1, 2, 2};
    double tensor1[36];
    double tensor2[8];

    fill_deterministic_values(tensor1, 36, 123u);
    fill_deterministic_values(tensor2, 8, 777u);

    return run_convnd_case("rank4 larger deterministic", rank, dim1, tensor1, dim2, tensor2);
}

typedef int (*convnd_test_fn)(void);

typedef struct
{
    const char *key;
    convnd_test_fn fn;
} convnd_named_test;

static const convnd_named_test CONVND_TESTS[] = {
    {"rank3_scalar_kernel", test_convnd_rank3_scalar_kernel},
    {"rank3_rectangular", test_convnd_rank3_rectangular},
    {"rank4_mixed_values", test_convnd_rank4_mixed_values},
    {"rank5_sparse_pattern", test_convnd_rank5_sparse_pattern},
    {"rank6_high_rank", test_convnd_rank6_high_rank},
    {"rank3_zero_tensors", test_convnd_rank3_zero_tensors},
    {"rank4_identity_kernel", test_convnd_rank4_identity_kernel},
    {"rank5_identity_kernel", test_convnd_rank5_identity_kernel},
    {"rank3_large_deterministic", test_convnd_rank3_large_deterministic},
    {"rank4_large_deterministic", test_convnd_rank4_large_deterministic}};

static int convnd_test_count(void)
{
    return (int) (sizeof(CONVND_TESTS) / sizeof(CONVND_TESTS[0]));
}

static void print_convnd_test_names(void)
{
    for (int i = 0; i < convnd_test_count(); ++i)
    {
        printf("%s\n", CONVND_TESTS[i].key);
    }
}

static int run_convnd_test_by_name(const char *test_name)
{
    for (int i = 0; i < convnd_test_count(); ++i)
    {
        if (strcmp(test_name, CONVND_TESTS[i].key) == 0)
        {
            return CONVND_TESTS[i].fn();
        }
    }

    printf("[FAIL] unknown test name: %s\n", test_name);
    printf("Available tests:\n");
    print_convnd_test_names();
    return 1;
}

static int run_convnd_tests(void)
{
    int failures = 0;

    for (int i = 0; i < convnd_test_count(); ++i)
    {
        failures += CONVND_TESTS[i].fn();
    }

    return failures;
}


#endif  // TESTS_TEST_CONVOLUTION_H
