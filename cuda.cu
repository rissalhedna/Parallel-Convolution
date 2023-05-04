#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>

#define DEFAULT_ITERATIONS 1
#define KERNEL_DIM 5
#define KERNEL_SIZE 25
#define TILE_SIZE 2

__global__ void convolution(int *grid, int *result, int *kernel, int dim) {
    int num_pads = (KERNEL_DIM - 1) / 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int outputRow = row - num_pads;
    int outputCol = col - num_pads;

    int val = 0;

    for (int i = 0; i < KERNEL_DIM; ++i) {
        for (int j = 0; j < KERNEL_DIM; ++j) {
            int inputRow = outputRow + i;
            int inputCol = outputCol + j;
            
            if (inputRow >= 0 && inputRow < dim && inputCol >= 0 && inputCol < dim) {
                val += grid[inputRow * dim + inputCol] * kernel[i * KERNEL_DIM + j];
            }
        }
    }

    if (outputRow >= 0 && outputRow < dim && outputCol >= 0 && outputCol < dim) {
        result[outputRow * dim + outputCol] = val;
    }
}

int main(int argc, char **argv) {
    int num_iterations;
    int DIM;
    int GRID_WIDTH;

    num_iterations = DEFAULT_ITERATIONS;
    if (argc >= 2) {
        DIM = atoi(argv[1]);
        GRID_WIDTH = DIM * DIM;
        if (argc == 3) {
            num_iterations = atoi(argv[2]);
        }
    } else {
        printf("Invalid command line arguments");
        exit(-1);
    }

    int *main_grid = (int *)malloc(GRID_WIDTH * sizeof(int));
    memset(main_grid, 0, GRID_WIDTH * sizeof(int));
    // Initialize grid with a simple pattern
memset(main_grid, 0, GRID_WIDTH * sizeof(int));
for (int i = 0; i < GRID_WIDTH; i++) {
        main_grid[i] = 1;
    }


    int kernel[KERNEL_SIZE] = {1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1};

    int *grid_d, *result_d, *kernel_d;
    int *result = (int *)malloc(GRID_WIDTH * sizeof(int));

    cudaMalloc((void **)&grid_d, GRID_WIDTH * sizeof(int));
    cudaMalloc((void **)&result_d, GRID_WIDTH * sizeof(int));
    cudaMalloc((void **)&kernel_d, KERNEL_SIZE * sizeof(int));

    cudaMemcpy(grid_d, main_grid, GRID_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel, KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((DIM + blockSize.x - 1) / blockSize.x, (DIM + blockSize.y - 1) / blockSize.y);

    for (int iter = 0; iter < num_iterations; iter++) {
        convolution<<<gridSize, blockSize>>>(grid_d, result_d, kernel_d, DIM);
        cudaMemcpy(result, result_d, GRID_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

       // Swap the grid and result pointers
int *temp = main_grid;
main_grid = result;
result = temp;



    cudaMemcpy(grid_d, main_grid, GRID_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
}

printf("\nConvolution Output: \n");
// for (int i = 0; i < GRID_WIDTH; i++) {
//     if (i % DIM == 0) {
//         printf("\n");
//     }
//     printf("%d  ", main_grid[i]);
// }
// printf("\n");

// Cleanup
cudaFree(grid_d);
cudaFree(result_d);
cudaFree(kernel_d);
free(main_grid);
free(result);

return 0;
}