#define _POSIX_C_SOURCE 200809L

#include "src/convolution.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RANK       3
#define DEFAULT_D0 32
#define DEFAULT_D1 32
#define DEFAULT_D2 32

#define DEFAULT_K0 5
#define DEFAULT_K1 5
#define DEFAULT_K2 5

#define DEFAULT_ITERATIONS 100

static int tensor_size(const int dims[], const int rank)
{
    int size = 1;

    for (int i = 0; i < rank; ++i)
    {
        size *= dims[i];
    }

    return size;
}

static void fill_deterministic(double *values, const int length, unsigned int seed)
{
    unsigned int state = seed;

    for (int i = 0; i < length; ++i)
    {
        state = state * 1664525u + 1013904223u;
        values[i] = ((double) (state % 2001u) - 1000.0) / 1000.0;
    }
}

static double elapsed_seconds(const struct timespec start, const struct timespec end)
{
    const double sec = (double) (end.tv_sec - start.tv_sec);
    const double nsec = (double) (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    return sec + nsec;
}

static double max_abs_diff(const double *lhs, const double *rhs, const int length)
{
    double max_diff = 0.0;

    for (int i = 0; i < length; ++i)
    {
        const double diff = fabs(lhs[i] - rhs[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }

    return max_diff;
}

static void print_usage(const char *program)
{
    printf("Usage: %s [--iterations N] [--tensor D0,D1,D2] [--kernel K0,K1,K2]\n", program);
    printf("\n");
    printf("Options:\n");
    printf("  --iterations N       Number of benchmark iterations (default: %d)\n",
           DEFAULT_ITERATIONS);
    printf("  --tensor D0,D1,D2    Input tensor dimensions (default: %d,%d,%d)\n", DEFAULT_D0,
           DEFAULT_D1, DEFAULT_D2);
    printf("  --kernel K0,K1,K2    Kernel tensor dimensions (default: %d,%d,%d)\n", DEFAULT_K0,
           DEFAULT_K1, DEFAULT_K2);
    printf("  --help               Show this message\n");
}

static int parse_positive_int(const char *text, int *value)
{
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX)
    {
        return -1;
    }

    *value = (int) parsed;
    return 0;
}

static int parse_rank3_dims(const char *text, int dims[RANK])
{
    char buffer[128];
    const size_t len = strlen(text);
    char *token = NULL;
    char *saveptr = NULL;
    int index = 0;

    if (len == 0 || len >= sizeof(buffer))
    {
        return -1;
    }

    memcpy(buffer, text, len + 1);

    token = strtok_r(buffer, ",xX", &saveptr);
    while (token != NULL && index < RANK)
    {
        if (parse_positive_int(token, &dims[index]) != 0)
        {
            return -1;
        }

        ++index;
        token = strtok_r(NULL, ",xX", &saveptr);
    }

    if (index != RANK || token != NULL)
    {
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int iterations = DEFAULT_ITERATIONS;
    int dim1[RANK] = {DEFAULT_D0, DEFAULT_D1, DEFAULT_D2};
    int dim2[RANK] = {DEFAULT_K0, DEFAULT_K1, DEFAULT_K2};
    int res_dim_direct[RANK] = {0};
    int res_dim_plan[RANK] = {0};

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }

        if (strcmp(argv[i], "--iterations") == 0)
        {
            if (i + 1 >= argc || parse_positive_int(argv[++i], &iterations) != 0)
            {
                printf("Invalid value for --iterations\n");
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }

        if (strcmp(argv[i], "--tensor") == 0)
        {
            if (i + 1 >= argc || parse_rank3_dims(argv[++i], dim1) != 0)
            {
                printf("Invalid value for --tensor. Expected D0,D1,D2\n");
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }

        if (strcmp(argv[i], "--kernel") == 0)
        {
            if (i + 1 >= argc || parse_rank3_dims(argv[++i], dim2) != 0)
            {
                printf("Invalid value for --kernel. Expected K0,K1,K2\n");
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }

        printf("Unknown argument: %s\n", argv[i]);
        print_usage(argv[0]);
        return 1;
    }

    const int size1 = tensor_size(dim1, RANK);
    const int size2 = tensor_size(dim2, RANK);

    convnd_plan *plan = convnd_plan_create(RANK, dim1, dim2, res_dim_plan);
    if (plan == NULL)
    {
        printf("Plan creation failed.\n");
        return 1;
    }

    const int result_size = convnd_plan_result_size(plan);
    if (result_size <= 0)
    {
        printf("Invalid planned result size.\n");
        convnd_plan_destroy(plan);
        return 1;
    }

    double *tensor1 = (double *) malloc((size_t) size1 * sizeof(double));
    double *tensor2 = (double *) malloc((size_t) size2 * sizeof(double));
    double *direct_out = (double *) malloc((size_t) result_size * sizeof(double));
    double *plan_out = (double *) malloc((size_t) result_size * sizeof(double));

    if (tensor1 == NULL || tensor2 == NULL || direct_out == NULL || plan_out == NULL)
    {
        printf("Allocation failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    fill_deterministic(tensor1, size1, 123u);
    fill_deterministic(tensor2, size2, 987u);

    struct timespec t0;
    struct timespec t1;

    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0)
    {
        printf("clock_gettime failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        fill_deterministic(tensor1, size1, 123u + (unsigned int) iter);
        fill_deterministic(tensor2, size2, 987u + (unsigned int) iter);

        if (convnd(RANK, tensor1, dim1, tensor2, dim2, direct_out, res_dim_direct) != 0)
        {
            printf("Direct convnd failed at iteration %d.\n", iter);
            free(tensor1);
            free(tensor2);
            free(direct_out);
            free(plan_out);
            convnd_plan_destroy(plan);
            return 1;
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        printf("clock_gettime failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    const double direct_seconds = elapsed_seconds(t0, t1);

    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0)
    {
        printf("clock_gettime failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        fill_deterministic(tensor1, size1, 123u + (unsigned int) iter);
        fill_deterministic(tensor2, size2, 987u + (unsigned int) iter);

        if (convnd_plan_execute(plan, tensor1, tensor2, plan_out) != 0)
        {
            printf("Planned execution failed at iteration %d.\n", iter);
            free(tensor1);
            free(tensor2);
            free(direct_out);
            free(plan_out);
            convnd_plan_destroy(plan);
            return 1;
        }
    }

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0)
    {
        printf("clock_gettime failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    const double plan_seconds = elapsed_seconds(t0, t1);

    fill_deterministic(tensor1, size1, 123u + (unsigned int) (iterations - 1));
    fill_deterministic(tensor2, size2, 987u + (unsigned int) (iterations - 1));

    if (convnd(RANK, tensor1, dim1, tensor2, dim2, direct_out, res_dim_direct) != 0
        || convnd_plan_execute(plan, tensor1, tensor2, plan_out) != 0)
    {
        printf("Final correctness check failed.\n");
        free(tensor1);
        free(tensor2);
        free(direct_out);
        free(plan_out);
        convnd_plan_destroy(plan);
        return 1;
    }

    const double diff = max_abs_diff(direct_out, plan_out, result_size);

    printf("=== Plan vs Direct Timing ===\n");
    printf("Tensor dims: [%d, %d, %d]\n", dim1[0], dim1[1], dim1[2]);
    printf("Kernel dims: [%d, %d, %d]\n", dim2[0], dim2[1], dim2[2]);
    printf("Result dims: [%d, %d, %d]\n", res_dim_plan[0], res_dim_plan[1], res_dim_plan[2]);
    printf("Iterations:  %d\n", iterations);
    printf("Direct total:   %.6f s (%.3f ms/iter)\n", direct_seconds,
           (direct_seconds * 1000.0) / iterations);
    printf("Planned total:  %.6f s (%.3f ms/iter)\n", plan_seconds,
           (plan_seconds * 1000.0) / iterations);

    if (plan_seconds > 0.0)
    {
        printf("Speedup: %.2fx\n", direct_seconds / plan_seconds);
    }
    else
    {
        printf("Speedup: n/a\n");
    }

    printf("Max |direct - planned| on final iteration: %.12f\n", diff);

    free(tensor1);
    free(tensor2);
    free(direct_out);
    free(plan_out);
    convnd_plan_destroy(plan);
    return 0;
}
