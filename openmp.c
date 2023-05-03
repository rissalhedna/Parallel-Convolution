#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define DEFAULT_ITERATIONS 1

#define KERNEL_DIM 5
#define KERNEL_SIZE 25

int conv_column(int *sub_grid, int i, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int counter = 0;
    int num_pads = (ke rnel_dim - 1) / 2;

    for (int j = 1; j < (num_pads + 1); j++)
    {
        counter += sub_grid[i + j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) + j * kernel_dim];
        counter += sub_grid[i - j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) - j * kernel_dim];
    }
    counter += sub_grid[i] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2)];

    return counter;
}

void convolve_omp(int *sub_grid, int *new_grid, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int num_pads = (kernel_dim - 1) / 2;

#pragma omp parallel for schedule(static)
    for (int i = num_pads * DIM; i < (DIM * (num_pads + nrows)); i++)
    {
        new_grid[i - (num_pads * DIM)] = conv_column(sub_grid, i, nrows, DIM, kernel, kernel_dim);
    }
}

int *check_omp(int *sub_grid, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int *new_grid = calloc(DIM * nrows, sizeof(int));

    convolve_omp(sub_grid, new_grid, nrows, DIM, kernel, kernel_dim);

    return new_grid;
}
int main(int argc, char **argv)
{
    int num_iterations = DEFAULT_ITERATIONS;
    int DIM = 0, GRID_WIDTH = 0, KERNEL_DIM = 0, KERNEL_SIZE = 0;

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
        printf("Invalid command line arguments\n");
        exit(-1);
    }

    int *main_grid = calloc(GRID_WIDTH, sizeof(int));
    assert(main_grid != NULL);
    memset(main_grid, 1, GRID_WIDTH * sizeof(int));

    int num_pads = (KERNEL_DIM - 1) / 2;

    int *kernel = calloc(KERNEL_SIZE, sizeof(int));
    assert(kernel != NULL);
    memset(kernel, 1, KERNEL_SIZE * sizeof(int));

    int *padded_grid = calloc((DIM + (num_pads * 2)) * (DIM + (num_pads * 2)), sizeof(int));
    assert(padded_grid != NULL);
    memcpy(&padded_grid[num_pads * (DIM + 2)], main_grid, GRID_WIDTH * sizeof(int));

    int *new_grid = calloc(GRID_WIDTH, sizeof(int));
    assert(new_grid != NULL);

    struct timeval start_time, end_time;
    double elapsed_time;

    gettimeofday(&start_time, NULL);

    omp_set_num_threads(4);

    for (int iter = 0; iter < num_iterations; iter++)
    {
        check_omp(padded_grid, DIM, DIM + (num_pads * 2), kernel, KERNEL_DIM);
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
    free(kernel);
    free(main_grid);

    return 0;
}
