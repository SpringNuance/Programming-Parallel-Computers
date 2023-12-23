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

void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> mean(ny);
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        double sumRow = 0;
        for(int x = 0; x < nx; x++){
            sumRow += data[x + y*nx];
        }
        mean[y] = sumRow / nx;
    }


    std::vector<double> difference(ny*nx);
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        for(int x = 0; x < nx; x++){
            difference[x + y*nx] = data[x + y*nx] - mean[y];
        }
    }

    std::vector<double> correlation(ny*nx);
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        double sum = 0;
        for(int x = 0; x < nx; x++){
            sum += difference[x + y*nx] * difference[x + y*nx];
        }
        double sqrtSum = sqrt(sum);
        for(int x = 0; x < nx; x++){
            correlation[x + y * nx] = difference[x + y *nx]/sqrtSum;
        }   
    }

    #pragma omp parallel for schedule(static, 1)
    for(int i = 0; i < ny; i++){
        for(int j = 0; j <= i; j++){
            double sum = 0;
            for(int x = 0; x < nx; x++){
                sum += correlation[x + i*nx] * correlation[x + j*nx];
            }
            result[i + j * ny] = sum;
        }
    }   
}

// .ppc/grader.py test