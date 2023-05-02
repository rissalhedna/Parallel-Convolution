#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#define DEFAULT_ITERATIONS 1
#define KERNEL_DIM 5
#define KERNEL_SIZE 25
int conv_column(int *, int, int, int, int *, int);
int conv(int *, int, int, int, int *, int);
int *check(int *, int, int, int *, int);

int conv_column(int *sub_grid, int i, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int counter = 0;
    int num_pads = (kernel_dim - 1) / 2;

    for (int j = 1; j < (num_pads + 1); j++)
    {
        counter = counter + sub_grid[i + j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) + j * kernel_dim];
        counter = counter + sub_grid[i - j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) - j * kernel_dim];
    }
    counter = counter + sub_grid[i] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2)];

    return counter;
}

int conv(int *sub_grid, int i, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int counter = 0;
    int num_pads = (kernel_dim - 1) / 2;
    // convolve middle column
    counter = counter + conv_column(sub_grid, i, nrows, DIM, kernel, kernel_dim);

    // convolve left and right columns
    for (int j = 1; j < (num_pads + 1); j++)
    {
        // get last element of current row
        int end = (((i / DIM) + 1) * DIM) - 1;
        if (i + j - end <= 0)
        { // if column is valid
            counter = counter + conv_column(sub_grid, i + j, nrows, DIM, kernel, kernel_dim);
        }
        // get first element of current row
        int first = (i / DIM) * DIM;
        if (i - j - first >= 0)
        {
            counter = counter + conv_column(sub_grid, i - j, nrows, DIM, kernel, kernel_dim);
        }
    }

    return counter;
}

int *check(int *sub_grid, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int val;
    int num_pads = (kernel_dim - 1) / 2;
    int *new_grid = calloc(DIM * nrows, sizeof(int));
    for (int i = (num_pads * DIM); i < (DIM * (num_pads + nrows)); i++)
    {
        val = conv(sub_grid, i, nrows, DIM, kernel, kernel_dim);
        new_grid[i - (num_pads * DIM)] = val;
    }
    return new_grid;
}
int main(int argc, char **argv)
{
    int iters = 0;
    int num_iterations;
    int DIM;
    int GRID_WIDTH;

    num_iterations = DEFAULT_ITERATIONS;
    if (argc >= 2)
    {
        DIM = atoi(argv[1]);
        GRID_WIDTH = DIM * DIM;
        if (argc == 3)
        {
            num_iterations = atoi(argv[2]);
        }
    }
    else
    {
        printf("Invalid command line arguments");
        exit(-1);
    }
    int main_grid[GRID_WIDTH];
    memset(main_grid, 0, GRID_WIDTH * sizeof(int));
    for (int i = 0; i < GRID_WIDTH; i++)
    {
        main_grid[i] = 1;
    }

    int num_pads = (KERNEL_DIM - 1) / 2;

    // memset(kernel, 0, KERNEL_SIZE * sizeof(int));
    // for (int i = 0; i < KERNEL_SIZE; i++)
    // {
    int kernel[KERNEL_SIZE] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};
    // }

    for (iters = 0; iters < num_iterations; iters++)
    {
        int upper[DIM * num_pads];
        int lower[DIM * num_pads];

        int *pad_row_upper;
        int *pad_row_lower;

        pad_row_upper = upper;
        pad_row_lower = lower;

        int start = 0;
        int end = DIM - 1;
        int nrows = end + 1 - start;

        int sub_grid[DIM * (nrows + (2 * num_pads))];

        memcpy(sub_grid, pad_row_upper, sizeof(int) * DIM * num_pads);
        memcpy(&sub_grid[DIM * num_pads], &main_grid[DIM * start], sizeof(int) * DIM * nrows);
        memcpy(&sub_grid[DIM * (nrows + num_pads)], pad_row_lower, sizeof(int) * DIM * num_pads);
        int *changed_subgrid = check(sub_grid, nrows, DIM, kernel, KERNEL_DIM);

        for (int i = 0; i < nrows * DIM; i++)
        {
            main_grid[i] = changed_subgrid[i];
        }
        // Output the updated grid state
        // if ( ID == 0 ) {
        // printf("\nConvolution Output: \n");
        // for (int j = 0; j < GRID_WIDTH; j++)
        // {
        //     if (j % DIM == 0)
        //     {
        //         printf("\n");
        //     }
        //     printf("%d  ", main_grid[j]);
        // }
        // printf("\n");
        // }

        free(changed_subgrid);
    }

    return 0;
}
