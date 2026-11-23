# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>ENTER YOUR NAME: DHARANI.A</h3>
<h3>ENTER YOUR REGISTER NO: 212225040067</h3>
<h3>EX. NO:04</h3>
<h3>DATE:21.11.2026</h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
TYPE YOUR CODE HERE

```
%%writefile matrix_add_unified_memset_compare.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, code:%d, reason:%s\n", \
            __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

void initialData(float *ip, int size)
{
    for (int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

__global__ void sumMatrixGPU(float *A, float *B, float *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny)
    {
        int idx = iy * nx + ix;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    printf("Unified Memory Matrix Addition: memset vs no memset\n");

    int nx = 1 << 12;  
    int ny = 1 << 12;
    int n = nx * ny;
    size_t nBytes = n * sizeof(float);

    float *A, *B, *C1, *C2;
    CHECK(cudaMallocManaged(&A, nBytes));
    CHECK(cudaMallocManaged(&B, nBytes));
    CHECK(cudaMallocManaged(&C1, nBytes)); // WITH memset
    CHECK(cudaMallocManaged(&C2, nBytes)); // WITHOUT memset

    initialData(A, n);
    initialData(B, n);

    dim3 block(32, 32);
    dim3 grid((nx + 31) / 32, (ny + 31) / 32);

    // ---------------- WITH MEMSET ----------------
    CHECK(cudaMemset(C1, 0, nBytes));

    double start = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, C1, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double time_with = seconds() - start;

    printf("GPU Time WITH memset: %f sec\n", time_with);

    // ---------------- WITHOUT MEMSET ----------------
    // NOTE: C2 is NOT memset → reused memory

    start = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, C2, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double time_without = seconds() - start;

    printf("GPU Time WITHOUT memset: %f sec\n", time_without);

    // Cleanup
    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(C1));
    CHECK(cudaFree(C2));

    CHECK(cudaDeviceReset());
    return 0;
}

!nvcc -arch=sm_75 matrix_add_unified_memset_compare.cu -o unified_test
!./unified_test
```
## OUTPUT:
SHOW YOUR OUTPUT HERE
https://youtu.be/L7hSX1DUN-A?si=CoZOEYe69Go2nGl-
<img width="491" height="78" alt="image" src="https://github.com/user-attachments/assets/ce8b08d0-559d-4052-a35d-62a97e0bea83" />

## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given 0.00967 time.
