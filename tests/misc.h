#ifndef TESTS_MISC_H
#define TESTS_MISC_H

void fill_mat(double *mat, const int length, const double val)
{
    for (int i = 0; i < length; ++i)
    {
        mat[i] = val;
    }
}

#endif  // TESTS_MISC_H