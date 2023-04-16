#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"
#include <omp.h>

#define DEFAULT_ITERATIONS 1

int conv_column(int *, int, int, int, int *, int);
int conv(int *, int, int, int, int *, int);
int * check(int *, int, int, int *, int);

int conv_column(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;

  for (int j = 1; j < (num_pads + 1); j++) {
    counter += sub_grid[i + j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) + j*kernel_dim];
    counter += sub_grid[i - j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) - j*kernel_dim];
  }
  counter += sub_grid[i] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2)];

  return counter;
}

int conv(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;
  //convolve middle column
  counter += conv_column(sub_grid, i, nrows, DIM, kernel, kernel_dim);

  //convolve left and right columns
  for (int j = 1; j < (num_pads + 1); j++) {
    //get last element of current row
    int end = (((i / DIM) + 1) * DIM) - 1;
    if (i + j - end <= 0) { //if column is valid
      counter += conv_column(sub_grid, i + j, nrows, DIM, kernel, kernel_dim);
    }
    //get first element of current row
    int first = (i / DIM) * DIM;
    if (i - j - first >= 0) {
      counter += conv_column(sub_grid, i - j, nrows, DIM, kernel, kernel_dim);
    }
  }

  return counter;
}

int * check(int * sub_grid, int nrows, int DIM, int * kernel, int kernel_dim) {
  int val;
  int num_pads = (kernel_dim - 1) / 2;
  int * new_grid = calloc(DIM * nrows, sizeof(int));

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = num_pads * DIM; i < (DIM * (num_pads + nrows)); i++) {
    for (int j = 0; j < DIM; j++) {
      int index = (i - (num_pads * DIM)) * DIM + j;
      val = conv(sub_grid, i + j, nrows, DIM, kernel, kernel_dim);
      new_grid[index] = val;
    }
  }
  return new_grid;
}

int main ( int argc, char** argv ) {
  int iters = 0;
  int num_iterations;
  int DIM;
  int GRID_WIDTH;
  int KERNEL_DIM;
  int KERNEL_SIZE;

  num_iterations = DEFAULT_ITERATIONS;
  if (argc >= 3) {
    DIM = atoi(argv[1]);
    GRID_WIDTH = DIM * DIM;
    KERNEL_DIM = atoi(argv[2]);
    KERNEL_SIZE = KERNEL_DIM * KERNEL_DIM;
    if (argc == 4) {
      num_iterations = atoi(argv[3]);
    }
  } else {
    printf("Invalid command line arguments");
    exit(-1);
  }

  int * main_grid = calloc(GRID_WIDTH, sizeof(int));
  for(int i = 0; i < GRID_WIDTH; i++) {
    main_grid[i] = 1;
  }

  int num_pads = (KERNEL_DIM - 1) / 2;

  int * kernel = calloc(KERNEL_SIZE, sizeof(int));
  for(int i = 0; i < KERNEL_SIZE; i++) {
    kernel[i] = 1;
  }

#pragma omp parallel
  {
    // OpenMP variables
    int ID, num_threads;
    int upper[DIM * num_pads];
    int lower[DIM * num_pads];

    int * pad_row_upper;
    int * pad_row_lower;

    // OpenMP setup
    ID = omp_get_thread_num();
    num_threads = omp_get_num_threads();

    assert(DIM % num_threads == 0);

    int start = (DIM / num_threads) * ID;
    int end = (DIM / num_threads) - 1 + start;
    int nrows = end + 1 - start;
    int next = (ID + 1) % num_threads;
    int prev = ID != 0 ? ID - 1 : num_threads - 1;

    for ( iters = 0; iters < num_iterations; iters++ ) {

      memcpy(lower, &main_grid[DIM * (end - num_pads + 1)], sizeof(int) * DIM * num_pads);
      pad_row_lower = malloc(sizeof(int) * DIM * num_pads);

      memcpy(upper, &main_grid[DIM * start], sizeof(int) * DIM * num_pads);
      pad_row_upper = malloc(sizeof(int) * DIM * num_pads);

      if(num_threads > 1) {
        if(ID % 2 == 1) {
          memcpy(pad_row_lower, &main_grid[DIM * (end + 1)], sizeof(int) * DIM * num_pads);
          memcpy(pad_row_upper, &main_grid[DIM * (start - num_pads)], sizeof(int) * DIM * num_pads);
        } else {
          memcpy(&main_grid[DIM * (end + 1)], lower, sizeof(int) * DIM * num_pads);
          memcpy(&main_grid[DIM * (start - num_pads)], upper, sizeof(int) * DIM * num_pads);
        }
        if(ID % 2 == 1) {
          memcpy(&main_grid[DIM * start], upper, sizeof(int) * DIM * num_pads);
          memcpy(&main_grid[DIM * (end + 1 - num_pads)], lower, sizeof(int) * DIM * num_pads);
        } else {
          memcpy(lower, &main_grid[DIM * (end - num_pads + 1)], sizeof(int) * DIM * num_pads);
          memcpy(upper, &main_grid[DIM * start], sizeof(int) * DIM * num_pads);
        }
      } else {
        pad_row_lower = upper;
        pad_row_upper = lower;
      }

      int sub_grid[DIM * (nrows + (2 * num_pads))];
      if (ID == 0) {
        memset(pad_row_upper, 0, DIM*sizeof(int)*num_pads);
      }
     
      if (ID == (num_threads - 1)) {
        memset(pad_row_lower, 0, DIM*sizeof(int)*num_pads);
      }
      memcpy(sub_grid, pad_row_upper, sizeof(int) * DIM * num_pads);
      memcpy(&sub_grid[DIM * num_pads], &main_grid[DIM * start], sizeof(int) * DIM * nrows);
      memcpy(&sub_grid[DIM * (nrows + num_pads)], pad_row_lower, sizeof(int) * DIM * num_pads);

      int * changed_subgrid = check(sub_grid, nrows, DIM, kernel, KERNEL_DIM);

      if(ID != 0) {
#pragma omp critical
        {
          MPI_Send(changed_subgrid, nrows * DIM, MPI_INT, 0, 11, MPI_COMM_WORLD);
          MPI_Recv(&main_grid[DIM * start], nrows * DIM, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
        }
      } else {
        for(int i = 0; i < nrows * DIM; i++) {
          main_grid[i] = changed_subgrid[i];
        }

        for(int k = 1; k < num_threads; k++) {
          MPI_Recv(&main_grid[DIM * (DIM / num_threads) * k], nrows * DIM, MPI_INT, k, 11, MPI_COMM_WORLD, &status);
        }

        for(int i = 1; i < num_threads; i++) {
          MPI_Send(main_grid, DIM * DIM, MPI_INT, i, 10, MPI_COMM_WORLD);
        }

      }

      free(pad_row_upper);
      free(pad_row_lower);
      free(changed_subgrid);
#pragma omp barrier
    }
  }

  // Output the updated grid state
  printf("\nConvolution Output: \n");
  for (int j = 0; j < GRID_WIDTH; j++) {
    if (j % DIM == 0) {
      printf("\n");
    }
    printf("%d  ", main_grid[j]);
  }
  printf("\n");

  free(main_grid);
  free(kernel);

  MPI_Finalize(); // finalize so I can exit
}
