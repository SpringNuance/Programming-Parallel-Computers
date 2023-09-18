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

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> mean(ny);
    for (int y = 0; y < ny; y++)
    {
        double sumRow = 0;
        for (int x = 0; x < nx; x++)
        {
            sumRow += data[x + y * nx];
        }
        mean[y] = sumRow / nx;
    }

    std::vector<double> difference(ny * nx);
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            difference[x + y * nx] = data[x + y * nx] - mean[y];
        }
    }

    std::vector<double> correlation(ny * nx);
    std::vector<double> correlationTranspose(nx * ny);
    for (int y = 0; y < ny; y++)
    {
        double sum = 0;
        for (int x = 0; x < nx; x++)
        {
            sum += difference[x + y * nx] * difference[x + y * nx];
        }
        double sqrtSum = sqrt(sum);
        for (int x = 0; x < nx; x++)
        {
            double cor = difference[x + y * nx] / sqrtSum;
            correlation[x + y * nx] = cor;
            correlationTranspose[y + x * ny] = cor;
        }
    }
    
    for (int y = 0; y < ny; y++){
        for (int x = 0; x < nx; x++){
            std::cout << correlation[x + y * nx] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int x = 0; x < nx; x++){
        for (int y = 0; y < ny; y++){
            std::cout << correlationTranspose[y + x * ny] << " ";
        }
        std::cout << std::endl;
    }
    

    int nny = roundup(ny, 4);
    int nnx = roundup(nx, 4);
    std::vector<double> correlationTransposeGPU(nx * nny);
    for (int x = 0; x < nx; x++){
        for (int y = 0; y < nny; y++){
            if (y < ny){
                correlationTransposeGPU[nny * x + y] = correlationTranspose[ny * x + y];
            } else {
                correlationTransposeGPU[nny * x + y] = 0.0;
            }
        }
    }
    
    std::cout << std::endl;
    for (int x = 0; x < nx; x++){
        for (int y = 0; y < nny; y++){
            std::cout << correlationTransposeGPU[nny * x + y] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < ny; i++){
        for (int j = 0; j <= i; j++){
            double sum = 0;
            for (int x = 0; x < nx; x++){
                sum += correlation[x + i * nx] * correlation[x + j * nx];
                // sum += correlationTranspose[i + x*ny] * correlationTranspose[j + x*ny];
            }
            result[i + j * ny] = sum;
        }
    }
}


// .ppc/grader.py test
// ./grading test-plain tests/032-small-simple.txt