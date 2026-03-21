#include "src/convolution.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_RANK 3
#define INPUT_D0 8
#define INPUT_D1 8
#define INPUT_D2 3

#define FILTER_RANK 3
#define FILTER_D0 3
#define FILTER_D1 3
#define FILTER_D2 3

static int tensor_size(const int dims[], const int rank)
{
    int size = 1;

    for (int i = 0; i < rank; ++i)
    {
        size *= dims[i];
    }

    return size;
}

static void fill_const(double *values, const int length, const double val)
{
    for (int i = 0; i < length; ++i)
    {
        values[i] = val;
    }
}

static void fill_random(double *values, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        values[i] = (rand() / (RAND_MAX + 1.0)) - 0.5;
    }
}

static void relu(double *values, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        if (values[i] < 0.0)
        {
            values[i] = 0.0;
        }
    }
}

static void add_bias(double *values, const int length, const double bias)
{
    for (int i = 0; i < length; ++i)
    {
        values[i] += bias;
    }
}

static void print_tensor_stats(const char *label, const double *values, const int length)
{
    double min_val = values[0];
    double max_val = values[0];
    double sum = 0.0;

    for (int i = 0; i < length; ++i)
    {
        if (values[i] < min_val)
        {
            min_val = values[i];
        }
        if (values[i] > max_val)
        {
            max_val = values[i];
        }
        sum += values[i];
    }

    const double mean = sum / length;

    printf("%-20s | Size: %8d | Min: %8.4f | Max: %8.4f | Mean: %8.4f\n",
           label,
           length,
           min_val,
           max_val,
           mean);
}

int main(void)
{
    srand(42);

    /*
     * Simulate a simple CNN inference:
     *   Input: 8x8x3 (height x width x channels)
     *   Conv filter: 3x3x3 (height x width x input_channels)
     *   Output: 6x6x1 per filter (height-2 x width-2 x 1)
     */

    int input_dim[INPUT_RANK] = {INPUT_D0, INPUT_D1, INPUT_D2};
    int filter_dim[FILTER_RANK] = {FILTER_D0, FILTER_D1, FILTER_D2};
    int output_dim[INPUT_RANK] = {0};

    const int input_size = tensor_size(input_dim, INPUT_RANK);
    const int filter_size = tensor_size(filter_dim, FILTER_RANK);
    const int output_size = (INPUT_D0 + FILTER_D0 - 1) * 
                            (INPUT_D1 + FILTER_D1 - 1) * 
                            (INPUT_D2 + FILTER_D2 - 1);

    double *input = (double *)malloc((size_t)input_size * sizeof(double));
    double *filter = (double *)malloc((size_t)filter_size * sizeof(double));
    double *output = (double *)malloc((size_t)output_size * sizeof(double));

    if (input == NULL || filter == NULL || output == NULL)
    {
        printf("Allocation failed.\n");
        free(input);
        free(filter);
        free(output);
        return 1;
    }

    printf("=== CNN Inference Example ===\n\n");
    printf("Input shape:  [%d, %d, %d]\n", INPUT_D0, INPUT_D1, INPUT_D2);
    printf("Filter shape: [%d, %d, %d]\n", FILTER_D0, FILTER_D1, FILTER_D2);

    fill_random(input, input_size);
    fill_random(filter, filter_size);

    printf("\n--- Before Convolution ---\n");
    print_tensor_stats("Input", input, input_size);
    print_tensor_stats("Filter", filter, filter_size);

    if (convnd(INPUT_RANK, input, input_dim, filter, filter_dim, output, output_dim) != 0)
    {
        printf("Convolution failed.\n");
        free(input);
        free(filter);
        free(output);
        return 1;
    }

    printf("\n--- After Convolution ---\n");
    printf("Output shape: [%d, %d, %d]\n", output_dim[0], output_dim[1], output_dim[2]);
    print_tensor_stats("Output (pre-relu)", output, output_size);

    const double bias = 0.1;
    add_bias(output, output_size, bias);

    printf("\n--- After Bias (bias=%.2f) ---\n", bias);
    print_tensor_stats("Output (post-bias)", output, output_size);

    relu(output, output_size);

    printf("\n--- After ReLU Activation ---\n");
    print_tensor_stats("Output (post-relu)", output, output_size);

    printf("\n=== Inference Complete ===\n");

    free(input);
    free(filter);
    free(output);
    return 0;
}
