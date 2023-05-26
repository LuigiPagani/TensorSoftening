#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 512
#define SOFTENING_RANGE 2
#define BLOCK_SIZE 8


void printMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        printf("Layer %d:\n", i);
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                printf("%d ", matrix[i * size * size + j * size + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void gpu_matrix_softening(int *a, int *c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int depth = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < N && column < N && depth < N) {
        int local_acc = 0;
        int value_sampled = 0;

        for (int i = -SOFTENING_RANGE; i < SOFTENING_RANGE + 1; i++) {
            if (row + i > -1 && row + i < N) {
                for (int j = -SOFTENING_RANGE; j < SOFTENING_RANGE + 1; j++) {
                    if (column + j > -1 && column + j < N) {
                        for (int k = -SOFTENING_RANGE; k < SOFTENING_RANGE + 1; k++) {
                            if (depth + k > -1 && depth + k < N) {
                                local_acc += a[(row + i) * N * N + (column + j) * N + (depth + k)];
                                value_sampled++;
                            }
                        }
                    }
                }
            }
        }
        local_acc = local_acc / value_sampled;
        c[row * N * N + column * N + depth] = local_acc;
    }
}

void Verifier(int *b, int *c) {
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int column = 0; column < MATRIX_SIZE; column++) {
            for (int depth = 0; depth < MATRIX_SIZE; depth++) {
                if (b[row * MATRIX_SIZE * MATRIX_SIZE + column * MATRIX_SIZE + depth] !=
                    c[row * MATRIX_SIZE * MATRIX_SIZE + column * MATRIX_SIZE + depth]) {
                    printf("Computation incorrect\n");
                    return;
                }
            }
        }
    }
    printf("Computation correct\n");
    return;
}

void cpu_matrix_softening(int *a, int *b) {
    int row, column, depth;
    for (row = 0; row < MATRIX_SIZE; row++) {
        for (column = 0; column < MATRIX_SIZE; column++) {
            for (depth = 0; depth < MATRIX_SIZE; depth++) {
                int local_acc = 0;
                int value_sampled = 0;

                for (int i = -SOFTENING_RANGE; i < SOFTENING_RANGE + 1; i++) {
                    if (row + i > -1 && row + i < MATRIX_SIZE) {
                        for (int j = -SOFTENING_RANGE; j < SOFTENING_RANGE + 1; j++) {
                            if (column + j > -1 && column + j < MATRIX_SIZE) {
                                for (int k = -SOFTENING_RANGE; k < SOFTENING_RANGE + 1; k++) {
                                    if (depth + k > -1 && depth + k < MATRIX_SIZE) {
                                        local_acc += a[(row + i) * MATRIX_SIZE * MATRIX_SIZE + (column + j) * MATRIX_SIZE + (depth + k)];
                                        value_sampled++;
                                    }
                                }
                            }
                        }
                    }
                }
                local_acc = local_acc / value_sampled;
                b[row * MATRIX_SIZE * MATRIX_SIZE + column * MATRIX_SIZE + depth] = local_acc;
            }
        }
    }
}

int main(int argc, char const *argv[]) {
    /// retrive some info about the CUDA device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  max Blocks Per MultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  num SM: %d\n", prop.multiProcessorCount);
        printf("  num bytes sharedMem Per Block: %d\n", prop.sharedMemPerBlock);
        printf("  num bytes sharedMem Per Multiprocessor: %d\n", prop.sharedMemPerMultiprocessor);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    int *a, *b, *c;
    a = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE);
    b = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE);
    c = (int *)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE);

    // initialize matrix A
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            for (int k = 0; k < MATRIX_SIZE; k++) {
                a[i * MATRIX_SIZE * MATRIX_SIZE + j * MATRIX_SIZE + k] = ((int)rand()) % 32;
            }
        }
    }

    // sequential version of matrix multiplication
    clock_t begin = clock();
    cpu_matrix_softening(a, b);
    clock_t end = clock();
    double time_spent = ((double)((end - begin)) * 1000) / CLOCKS_PER_SEC;
    printf("Time elapsed on naive CPU matrix softening of %dx%dx%d = %f ms\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, time_spent);

    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int N = MATRIX_SIZE;
    size_t size = N * N * N * sizeof(int);

    int *d_a, *d_c;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_c, size);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, (N + blockSize.z - 1) / blockSize.z);

    gpu_matrix_softening<<<gridSize, blockSize>>>(d_a, d_c, N);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on GPU matrix softening of %dx%dx%d = %f ms.\n\n", MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, gpu_elapsed_time_ms);

  /*
    printf("Matrix b (CPU result):\n");
    printMatrix(b, MATRIX_SIZE);
    printf("Matrix c (GPU result):\n");
    printMatrix(c, MATRIX_SIZE);
*/

    Verifier(b,c);

    return 0;
}
