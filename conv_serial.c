#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#define DEFAULT_ITERATIONS 1

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
    int num_iterations = DEFAULT_ITERATIONS;
    int DIM = 0, GRID_WIDTH = 0, KERNEL_DIM = 0, KERNEL_SIZE = 0;

    if (argc >= 3)
    {
        DIM = atoi(argv[1]);
        GRID_WIDTH = DIM * DIM;
        KERNEL_DIM = atoi(argv[2]);
        KERNEL_SIZE = KERNEL_DIM * KERNEL_DIM;

        if (argc == 4)
        {
            num_iterations = atoi(argv[3]);
        }
    }
    else
    {
        printf("Invalid command line arguments\n");
        exit(-1);
    }

    int main_grid[GRID_WIDTH];
    memset(main_grid, 0, GRID_WIDTH * sizeof(int));

    for (int i = 0; i < GRID_WIDTH; i++)
    {
        main_grid[i] = 1;
    }

    int num_pads = (KERNEL_DIM - 1) / 2;

    int kernel[KERNEL_SIZE];
    memset(kernel, 0, KERNEL_SIZE * sizeof(int));

    for (int i = 0; i < KERNEL_SIZE; i++)
    {
        kernel[i] = 1;
    }

    int *padded_grid = calloc((DIM + (num_pads * 2)) * (DIM + (num_pads * 2)), sizeof(int));
    memcpy(&padded_grid[num_pads * (DIM + 2)], main_grid, GRID_WIDTH * sizeof(int));

    int *new_grid = calloc(GRID_WIDTH, sizeof(int));

    struct timeval start_time, end_time;
    double elapsed_time;

    gettimeofday(&start_time, NULL);

    for (int iter = 0; iter < num_iterations; iter++)
    {
        new_grid = check(padded_grid, DIM, DIM + (num_pads * 2), kernel, KERNEL_DIM);

        // swap pointers to avoid copying values
        int *tmp = padded_grid;
        padded_grid = new_grid;
        new_grid = tmp;
    }

    gettimeofday(&end_time, NULL);
    elapsed_time = ((end_time.tv_sec - start_time.tv_sec) * 1000000u + end_time.tv_usec - start_time.tv_usec) / 1.e6;

    printf("Convolution of size %dx%d, kernel of size %dx%d, %d iterations took %lf seconds\n", DIM, DIM, KERNEL_DIM, KERNEL_DIM, num_iterations, elapsed_time);

    free(padded_grid);
    free(new_grid);

    return 0;
}

/*
DESCRIPTION:

1. The necessary header files are included - stdio.h, stdlib.h, assert.h, string.h, sys/time.h
2. The DEFAULT_ITERATIONS is defined as 1
3. Two functions for performing the convolution are defined: conv_column() and conv(). These functions perform convolution operations on the provided sub-grid based on a given kernel.
4. The check() function is defined. This function performs the convolution operation on the input sub-grid using the specified kernel and returns the resulting new grid.
5. In the main function, command line arguments are parsed to determine the size of the grid, kernel, and number of iterations to run.
6. The main grid is created and initialized to all 1's, while the kernel is created and initialized to all 1's.
7. The padded grid is created and initialized by copying the main grid into the center of it, with padding added to the edges.
8. A new grid is created to store the results of the convolution operation.
9. A timer is started to track the elapsed time of the convolution operation.
10. A loop runs for the specified number of iterations, calling the check() function to perform the convolution operation and storing the resulting new grid in the new_grid pointer.
11. Pointers are swapped to avoid copying values.
12. The timer is stopped and the elapsed time is printed.
13. Memory allocated using calloc() is freed.
14. The main function returns 0.
*/