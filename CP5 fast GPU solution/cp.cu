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
#include <cuda_runtime.h>
#include <vector>  
#include <chrono>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}


static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

/*
dim3 dimBlock(256, 1);
dim3 dimGrid(1, 1);
*/

__global__ void meanKernel(int nx, int ny, float* dataGPU, float* meanGPU) {
    int j = threadIdx.x;
    for (int c = 0; c < ny; c += 256) {
        int y = c + j;
        if (y >= ny) return;
        float sumRow = 0;
        for(int x = 0; x < nx; x++){
            sumRow += dataGPU[x + y*nx];
        }
        meanGPU[y] = sumRow / nx;
    }
}

/*
dim3 dimBlock(256, 1);
dim3 dimGrid(1, 256);
*/
__global__ void differenceKernel(int nx, int ny, float* dataGPU, float* meanGPU, float* differenceGPU) {
    int i = threadIdx.x;
    int j = blockIdx.y;
    for (int c = 0; c < ny; c += 256) {
        for (int d = 0; d < nx; d += 256) {
            int x = i + d;
            int y = j + c;
            if (x >= nx && y >= ny) return;
            if (x >= nx || y >= ny) continue;
            differenceGPU[x + y*nx] = dataGPU[x + y*nx] - meanGPU[y];
        }
    }
}

/*
dim3 dimBlock(256, 1);
dim3 dimGrid(1, 1);
*/
__global__ void correlationKernel(int nx, int ny, float* correlationTransposeOriginal, float* differenceGPU) {
    int j = threadIdx.x;

    for (int c = 0; c < ny; c += 256) {
        int y = c + j;
        if (y >= ny) return;
        float sum = 0;
        for(int x = 0; x < nx; x++){
            sum += differenceGPU[x + y*nx] * differenceGPU[x + y*nx];
        }
        float sqrtSum = sqrt(sum);
        for(int x = 0; x < nx; x++){
            double cor = differenceGPU[x + y * nx]/sqrtSum;
            correlationTransposeOriginal[y + x * ny] = cor;
        }   
    }
}

/*
dim3 dimBlock(64, 1);
dim3 dimGrid(1, nny);
*/
__global__ void paddingKernel(int nx, int ny, int nny, float* correlationTransposeOriginal, float* correlationTransposeGPU) {
    int x = threadIdx.x;
    int y = blockIdx.y;

    for (int c = 0; c < nx; c += 64) {
        int j = c + x;
        if (j >= nx) return;
        float v = (y < ny) ? correlationTransposeOriginal[ny * j + y] : 0.0;
        correlationTransposeGPU[nny * j + y] = v;
    }
}

/*
dim3 dimBlock(8, 8);
dim3 dimGrid(nny/64, nny/64);
*/
__global__ void resultKernel(int nx, int ny, int nny, float *resultGPU, float* correlationTransposeGPU){
    int iThread = threadIdx.x;
    int jThread = threadIdx.y;
    int iBlock = blockIdx.x;
    int jBlock = blockIdx.y;
    if (iBlock > jBlock) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int index_i = iBlock * 64 + i * 8 + iThread;
                int index_j = jBlock * 64 + j * 8 + jThread;
                if (index_i < ny && index_j < ny) {
                    resultGPU[ny * index_i + index_j] = 0;
                } 
            }
        }
    } else {
        float v[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                v[i][j] = 0.0;
            }
        }
        for (int k = 0; k < nx; k++) {
            float x[8];
            float y[8];
            for (int i = 0; i < 8; i++) {
                int index_i = iBlock * 64 + i * 8 + iThread;
                x[i] = correlationTransposeGPU[nny * k + index_i];
            }
            for (int j = 0; j < 8; j++) {
                int index_j = jBlock * 64 + j * 8 + jThread;
                y[j] = correlationTransposeGPU[nny * k + index_j];
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    v[i][j] += x[i] * y[j];
                }
            }
        }
        
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int index_i = iBlock * 64 + i * 8 + iThread;
                int index_j = jBlock * 64 + j * 8 + jThread;
                if (index_i < ny && index_j < ny) {
                    resultGPU[ny * index_i + index_j] = v[i][j];
                } 
            }
        }
    }
}


/* Main CPU side function */
void correlate(int ny, int nx, const float *data, float *result) {
    int nny = roundup(ny, 64);
    // Initializing the arrays
    float* dataGPU = NULL;
    float* meanGPU = NULL;
    float* differenceGPU = NULL;
    float* correlationTransposeOriginal = NULL;
    float* correlationTransposeGPU = NULL;
    float* resultGPU = NULL;
    // Allocating the data
    CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&meanGPU, ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&differenceGPU, ny * nx * sizeof(float)));
    CHECK(cudaMalloc((void**)&correlationTransposeOriginal, nx * ny * sizeof(float)));
    CHECK(cudaMalloc((void**)&correlationTransposeGPU, nx * nny * sizeof(float)));
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    // Copying the data
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    // Run mean kernel
    {
        dim3 dimBlock(256, 1);
        dim3 dimGrid(1, 1);
        meanKernel<<<dimGrid, dimBlock>>>(nx, ny, dataGPU, meanGPU);
        CHECK(cudaGetLastError());
    }
    
    // Run difference kernel
    {
        dim3 dimBlock(256, 1);
        dim3 dimGrid(1, 256);
        differenceKernel<<<dimGrid, dimBlock>>>(nx, ny, dataGPU, meanGPU, differenceGPU);
        CHECK(cudaGetLastError());
    }

    // Run correlation kernel
    {
        dim3 dimBlock(256, 1);
        dim3 dimGrid(1, 1);
        correlationKernel<<<dimGrid, dimBlock>>>(nx, ny, correlationTransposeOriginal, differenceGPU);
        CHECK(cudaGetLastError());
    }    

    // Run padding kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nny);
        paddingKernel<<<dimGrid, dimBlock>>>(nx, ny, nny, correlationTransposeOriginal, correlationTransposeGPU);
        CHECK(cudaGetLastError());
    }

    // Run calculating results kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nny/64, nny/64);
        resultKernel<<<dimGrid, dimBlock>>>(nx, ny, nny, resultGPU, correlationTransposeGPU);
        CHECK(cudaGetLastError());
    }

    cudaDeviceSynchronize();
    // Copy data back the results to the CPU side 
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    // Releasing all memory
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(meanGPU));
    CHECK(cudaFree(differenceGPU));
    CHECK(cudaFree(correlationTransposeOriginal));
    CHECK(cudaFree(correlationTransposeGPU));
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
// wget https://ppc-exercises.cs.aalto.fi/course/aalto2022/cp/cp5/cp5.zip unzip cp5.zip
// Finally, in the terminal, type 
// ./grading test
// for grading the tests

// ./grading test
// ./grading test-plain tests/001-small-raw.txt
// ./grading test-plain tests/120-medium-simple-memcheck.txt
// ./grading test-plain benchmarks/3.txt
// ./grading test-plain benchmarks/4a.txt
