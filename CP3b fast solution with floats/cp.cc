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
#include "vector.h"
#include <chrono>

void correlate(int ny, int nx, const float *data, float *result) {
    // elements per vector
    constexpr int nb = 8;
    // vectors per input row
    int na = (nx + nb - 1) / nb;
    // block size
    constexpr int nd = 9;
    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;


    // Constructing the vectorData consisting of an array of vectors of 4 floats from the data array
    // each row padded with 0s and there are rows padded with only 0s by size of nd = 3 rows
    // For example, if the data size is nx = 18, ny = 20
    // nd = 3 => 21 is the smallest number larger than 20 divisible by 3 => ncd = 21 and ncd - ny = 1. This one last row is filled with 0s
    // nb = 4 elements per vector => 20 is the smallest number larger than 18 divisible by 20 => na = 20/4 = 5. The last vector will have 2 elements padded by 0s
    // Data vector 
    float8_t* vectorData = float8_alloc(ncd*na);
    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        for (int ka = 0; ka < na; ka++) {
            for (int kb = 0; kb < nb; kb++) {
                int i = ka * nb + kb;
                vectorData[na*y + ka][kb] = i < nx ? data[nx * y + i] : 0.0;
            }
        }
    }
    
    #pragma omp parallel for
    for (int j = ny; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                vectorData[na*j + ka][kb] = 0.0;
            }
        }
    }

    // number of instruction level
    int ins = 3;
    int remainderInstruction = na % ins; 

    // Calculating the mean of each row. Mean of row y is stored in mean[y]
    // Difference vector
    float8_t vectorInstructionMean[ins * ny];
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        for (int m = 0; m < ins; m++){
            vectorInstructionMean[m + y*ins] = float8_0;
        }
    }
    std::vector<float> mean(ny);
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        float8_t sumVector = float8_0;
        for (int ka = 0; ka < na/ins; ka++){
            for (int m = 0; m < ins; m++){
                vectorInstructionMean[m + y*ins] += vectorData[ins * ka + m + y*na];
            }
        }
        for (int m = 0; m < ins; m++){
            sumVector += vectorInstructionMean[m + y*ins];
        }
        for (int ka = 1; ka <= remainderInstruction; ka++){  
            sumVector += vectorData[na - ka + y*na];
        }
        float sumRow = 0.0;
        for(int kb = 0; kb < nb; kb++){
            sumRow += sumVector[kb];
        }
        
        mean[y] = sumRow / nx;
    }


    // Calculating the difference between each element of the row and the mean of the row
    
    float8_t* difference = float8_alloc(ncd*na);
    int remainder = nx % nb;
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        for(int ka = 0; ka < na - 1; ka++){
            difference[na*y + ka] = vectorData[na*y + ka] - mean[y];
        }
        if (remainder == 0){
            difference[na*y + na - 1] = vectorData[na*y + na - 1] - mean[y];
        } else {
            for(int kb = 0; kb < nb; kb++){
                difference[na*y + na - 1][kb] = kb < remainder ? vectorData[na*y + na - 1][kb] - mean[y] : 0.0;
            }
        }        
    }
    // Padded rows are just 0s for difference vector
    #pragma omp parallel for
    for (int j = ny; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            difference[na*j + ka] = float8_0;
        }
    }

    // Calculating the correlation vector X
    

    float8_t vectorInstructionCorrelation[ins * ny];
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        for (int m = 0; m < ins; m++){
            vectorInstructionCorrelation[m + y*ins] = float8_0;
        }
    }

    float8_t* correlation = float8_alloc(ncd * na);
    #pragma omp parallel for
    for(int y = 0; y < ny; y++){
        float8_t sumVector = float8_0;
        for(int ka = 0; ka < na/ins; ka++){
            for (int m = 0; m < ins; m++){
                float8_t reuse = difference[ins * ka + m + na*y];
                vectorInstructionCorrelation[m + y*ins] += reuse * reuse ;
            }
        }
        for (int m = 0; m < ins; m++){
            sumVector += vectorInstructionCorrelation[m + y*ins];
        }
        for (int ka = 1; ka <= remainderInstruction; ka++){  
            float8_t reuse = difference[na - ka + y*na];
            sumVector += reuse * reuse;
        }

        float sumRow = 0.0;
        for(int kb = 0; kb < nb; kb++){
            sumRow += sumVector[kb];
        }

        for(int ka = 0; ka < na; ka++){
            correlation[na*y + ka] = difference[na*y + ka]/sqrt(sumRow);
        }   
    }

    // Padded rows are just 0s for correlation vector
    #pragma omp parallel for
    for (int j = ny; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            correlation[na*j + ka] = float8_0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Calculate the final XXT 
    #pragma omp parallel for schedule(dynamic, 1)
    for(int ic = 0; ic < nc; ic++){
        for(int jc = 0; jc <= ic; jc++){
            float8_t vv[nd][nd];
            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    vv[id][jd] = float8_0;
                }
            }
            //float8_t sumVector = float8_0;
            //float8_t s0 = float8_0;
            //float8_t s1 = float8_0;
            //float8_t s2 = float8_0;
            float8_t iRow[nd], jRow[nd];
            for(int ka = 0; ka < na; ka++){
                for (int kd = 0; kd < nd; kd++){
                    iRow[kd] = correlation[na*(ic * nd + kd) + ka];
                    jRow[kd] = correlation[na*(jc * nd + kd) + ka];
                }
                for (int id = 0; id < nd; ++id) {
                    for (int jd = 0; jd < nd; ++jd) {
                        vv[id][jd] += iRow[id] * jRow[jd];
                    }
                }
            }

            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    int i = ic * nd + id;
                    int j = jc * nd + jd;
                    float sumRow = 0;
                    float s0 = 0;
                    float s1 = 0;
                    for(int kb = 0; kb < nb/2; kb++){
                        s0 += vv[id][jd][kb * 2 + 0];
                        s1 += vv[id][jd][kb * 2 + 1];
                    }
                    if (i < ny && j < ny){
                        result[i + j*ny] = s0 + s1;
                    }
                }
            }
        }
    }    
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::free(difference);
    std::free(correlation);
    std::free(vectorData);
    //std::cout << "This code part takes " << (end - start).count() << "s\n";
}