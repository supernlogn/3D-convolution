#include "test_convolution.h"


int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        return run_convnd_tests() == 0 ? 0 : 1;
    }

    if (argc == 2 && strcmp(argv[1], "--list") == 0)
    {
        print_convnd_test_names();
        return 0;
    }

    if (argc == 2)
    {
        return run_convnd_test_by_name(argv[1]) == 0 ? 0 : 1;
    }

    printf("Usage: %s [--list|test_name]\n", argv[0]);
    return 1;
}