#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

#define DEFAULT_ITERATIONS 1

__global__ void convolve_cuda(int *sub_grid, int *new_grid, int nrows, int DIM, int *kernel, int kernel_dim)
{
    int num_pads = (kernel_dim - 1) / 2;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int i = row * DIM + col;

    if (row < nrows && col < nrows)
    {
        int counter = 0;

        for (int j = 1; j < (num_pads + 1); j++)
        {
            counter += sub_grid[i + j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) + j * kernel_dim];
            counter += sub_grid[i - j * DIM] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2) - j * kernel_dim];
        }
        counter += sub_grid[i] * kernel[(((kernel_dim - 1) * (kernel_dim + 1)) / 2)];

        new_grid[row * nrows + col] = counter;
    }
}

int *check_cuda(int *sub_grid, int nrows, int DIM, int *kernel, int kernel_dim, int num_iterations)
{
    int *new_grid = (int *)malloc(nrows * nrows * sizeof(int));
    int *d_sub_grid, *d_new_grid, *d_kernel;

    cudaMalloc(&d_sub_grid, (DIM + (kernel_dim - 1)) * (DIM + (kernel_dim - 1)) * sizeof(int));
    cudaMalloc(&d_kernel, kernel_dim * kernel_dim * sizeof(int));
    cudaMalloc(&d_new_grid, nrows * nrows * sizeof(int));

    cudaMemcpy(d_sub_grid, sub_grid, (DIM + (kernel_dim - 1)) * (DIM + (kernel_dim - 1)) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_dim * kernel_dim * sizeof(int), cudaMemcpyHostToDevice);j

    dim3 block_size(9, 9);
    dim3 num_blocks((nrows + block_size.x - 1) / block_size.x, (nrows + block_size.y - 1) / block_size.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < num_iterations; iter++)
    {
        convolve_cuda<<<num_blocks, block_size>>>(d_sub_grid, d_new_grid, nrows, DIM + (kernel_dim - 1), d_kernel, kernel_dim);

        int *tmp = d_sub_grid;
        d_sub_grid = d_new_grid;
        d_new_grid = tmp;
    }

   cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %f s\n", elapsed_time/100);

    cudaMemcpy(new_grid, d_sub_grid + ((kernel_dim - 1) / 2) * (DIM + (kernel_dim - 1)) + ((kernel_dim - 1) / 2), nrows * nrows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_sub_grid);
    cudaFree(d_kernel);

    return new_grid;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: ./convolve <DIM> <KERNEL_DIM> <NUM_ITERATIONS>\n");
        exit(1);
    }
    int DIM = atoi(argv[1]);
    int kernel_dim = atoi(argv[2]);
    int num_iterations = atoi(argv[3]);
    int *sub_grid = (int *)malloc((DIM + (kernel_dim - 1)) * (DIM + (kernel_dim - 1)) * sizeof(int));
    int *kernel = (int *)malloc(kernel_dim * kernel_dim * sizeof(int));

    srand(42);

    for (int i = 0; i < DIM + (kernel_dim - 1); i++)
    {
        for (int j = 0; j < DIM + (kernel_dim - 1); j++)
        {
            if (i < (kernel_dim - 1) / 2 || i >= DIM + (kernel_dim - 1) - (kernel_dim - 1) / 2 || j < (kernel_dim - 1) / 2 || j >= DIM + (kernel_dim - 1) - (kernel_dim - 1) / 2)
            {
                sub_grid[i * (DIM + (kernel_dim - 1)) + j] = 0;
            }
            else
            {
                sub_grid[i * (DIM + (kernel_dim - 1)) + j] = rand() % 100;
            }
        }
    }

    for (int i = 0; i < kernel_dim * kernel_dim; i++)
    {
        kernel[i] = rand() % 10;
    }

    int *result = check_cuda(sub_grid, DIM, DIM + (kernel_dim - 1), kernel, kernel_dim, num_iterations);

    // for (int i = 0; i < DIM; i++)
    // {
    //     for (int j = 0; j < DIM; j++)
    //     {
    //         printf("%d ", result[i * DIM + j]);
    //     }
    //     printf("\n");
    // }

    free(sub_grid);
    free(kernel);
    free(result);

    return 0;
}
