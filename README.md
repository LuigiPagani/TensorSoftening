# Matrix Softening

This code performs matrix softening using both CPU and GPU implementations. Matrix softening is a computation where each element in a three-dimensional matrix is replaced by the average value of its neighboring elements within a given range.

## Code Structure

The code consists of the following main components:

1. Header Files: The necessary header files are included for CUDA, standard input/output, standard library, and time functions.

2. Constants: Several constants are defined for the matrix size, softening range, and block size for GPU computation.

3. Utility Functions:
   - `printMatrix`: Prints the contents of a given matrix.
   - `Verifier`: Verifies if the GPU computation is correct by comparing the resulting matrix with the CPU computation.

4. CPU Implementation:
   - `cpu_matrix_softening`: Performs matrix softening using a sequential CPU implementation.

5. GPU Implementation:
   - `gpu_matrix_softening`: Performs matrix softening using a parallel GPU implementation using CUDA.

6. Main Function:
   - Retrieves information about the CUDA device(s).
   - Allocates memory for the matrices.
   - Initializes matrix A with random values.
   - Executes the CPU implementation and measures the elapsed time.
   - Executes the GPU implementation and measures the elapsed time.
   - Verifies the correctness of the GPU computation.

## Usage

1. Compilation: The code needs to be compiled using a CUDA-enabled compiler. Make sure to link against the CUDA runtime library.

2. Execution: Run the compiled binary file.

## Output

When the code is executed, it provides the following output:

1. CUDA Device Information: Displays information about the available CUDA devices, including device name, maximum blocks per multiprocessor, maximum threads per multiprocessor, maximum threads per block, number of streaming multiprocessors, shared memory per block, shared memory per multiprocessor, memory clock rate, memory bus width, and peak memory bandwidth.

2. CPU Matrix Softening Time: Displays the elapsed time for the sequential CPU matrix softening.

3. GPU Matrix Softening Time: Displays the elapsed time for the parallel GPU matrix softening using CUDA.

4. Computation Verification: Prints either "Computation correct" if the GPU computation matches the CPU computation or "Computation incorrect" if there is a mismatch.

Note: The code also includes commented-out code for printing the resulting matrices, which can be uncommented if desired.

## Additional Notes

- The code generates random values for matrix A using the `rand()` function. If you require deterministic results, you may need to modify the random number generation logic.

- The code uses three-dimensional matrices with a size of MATRIX_SIZE x MATRIX_SIZE x MATRIX_SIZE. Adjust the `MATRIX_SIZE` constant according to your requirements.

- The GPU implementation uses CUDA and parallelizes the computation using multiple threads and blocks. The block size can be adjusted by modifying the `BLOCK_SIZE` constant.

- The GPU implementation is executed using the CUDA runtime API functions and CUDA events to measure the execution time.

- The CPU implementation is executed sequentially on the CPU.

- The `printMatrix` function can be used to print the resulting matrices if desired. Uncomment the corresponding lines in the main function to enable this functionality.
