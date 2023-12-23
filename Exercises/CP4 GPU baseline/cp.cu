/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <math.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include <vector>

static inline void check(cudaError_t err, const char *context)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

/*
static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}
*/

__global__ void calculateResult(int nx, int ny, float *result, float *correlation)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny)
        return;
    float sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
        sum += correlation[x + i * nx] * correlation[x + j * nx];
    }
    result[i + j * ny] = sum;
}

/*
for(int i = 0; i < ny; i++){
    for(int j = 0; j <= i; j++){
        float sum = 0;
        for(int x = 0; x < nx; x++){
            sum += correlation[x + i*nx] * correlation[x + j*nx];
        }
        result[i + j * ny] = sum;
    }
}
*/

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<float> mean(ny);
    for (int y = 0; y < ny; y++)
    {
        float sumRow = 0;
        for (int x = 0; x < nx; x++)
        {
            sumRow += data[x + y * nx];
        }
        mean[y] = sumRow / nx;
    }

    std::vector<float> difference(ny * nx);
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            difference[x + y * nx] = data[x + y * nx] - mean[y];
        }
    }

    std::vector<float> correlation(ny * nx);
    for (int y = 0; y < ny; y++)
    {
        float sum = 0;
        for (int x = 0; x < nx; x++)
        {
            sum += difference[x + y * nx] * difference[x + y * nx];
        }
        float sqrtSum = sqrt(sum);
        for (int x = 0; x < nx; x++)
        {
            correlation[x + y * nx] = difference[x + y * nx] / sqrtSum;
        }
    }

    // Allocate memory & copy data to GPU
    /*
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));
    */
    float *correlationGPU = NULL;
    // correlation.data() return the pointers of the vectors
    CHECK(cudaMalloc((void **)&correlationGPU, ny * nx * sizeof(float)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void **)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(correlationGPU, correlation.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    /*
    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());
    */
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    calculateResult<<<dimGrid, dimBlock>>>(nx, ny, resultGPU, correlationGPU);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    /*
    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    */
    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(correlationGPU));
    CHECK(cudaFree(resultGPU));
}

// How to run GPU code in VS Code using Aalto remote computers
// First in the Remote Explorer, we click Add New, then type in this ssh:

// ssh nguyenb5@tavi -J nguyenb5@kosh.aalto.fi 

// where tavi is the Maari computer
// List of all computers
// https://www.aalto.fi/en/services/linux-computer-names-in-it-classrooms
// Replace nguyenb5 by your Aalto username

// Inpput the password when asked
// After being connected to remote computer in a new window, type in this command to download and unzip file
// wget https://ppc-exercises.cs.aalto.fi/course/aalto2022/cp/cp4/cp4.zip unzip cp4.zip
// Finally, in the terminal, type 
// ./grading test
// for grading the tests