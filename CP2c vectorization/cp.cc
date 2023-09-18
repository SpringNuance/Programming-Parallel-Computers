/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <iostream>
#include <math.h>
#include <vector>  
#include "vector.h"

void correlate(int ny, int nx, const float *data, float *result) {
    /*
    double4_t v1, v2;
    for (int k = 0; k < 4; k++){
        v1[k] = k;
        v2[k] = 2 * k;
    }
    double4_t v3 = v1 * v2 - 1.0;
    std::cout << "v3 is: ";
    for (int k = 0; k < 4; k++){
        std::cout << v3[k];
    }
    */
    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    int na = (nx + nb - 1) / nb;
    // Constructing the vectorData consisting of an array of vectors of 4 doubles from the data array, each row padded with 0s
    double4_t* vectorData = double4_alloc(ny*na);
    for (int r = 0; r < ny; r++) {
        for (int ka = 0; ka < na; ka++) {
            for (int kb = 0; kb < nb; kb++) {
                int s = ka * nb + kb;
                vectorData[na*r + ka][kb] = s < nx ? data[nx*r + s] : 0.0;
            }
        }
    }
    /* Printing the constructed vector of 4 from the data
    for (int r = 0; r < ny; r++) {
        for (int ka = 0; ka < na; ka++) {
            std::cout << "[";
            for (int kb = 0; kb < nb; kb++) {
                std::cout << " " << vectorData[na*r + ka][kb] << " ";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
    }
    */
    
    std::vector<double> meanj;
    for(int i = 0; i < ny; i++){
        double sumRowi = 0;
        for(int k = 0; k < nx; k++){
            sumRowi += data[k + i*nx];
        }
        double meanRowi = sumRowi / nx;
        meanj.push_back(meanRowi);
        for(int j = 0; j <= i; j++){
            double meanRowj = meanj[j];
            
            double nominator = 0;
            double sumi = 0;
            double sumj = 0;
            
            double4_t vectorMeani = double4_0;
            double4_t vectorMeanj = double4_0;

            double4_t conjugate_i = double4_0;
            double4_t conjugate_j = double4_0;
            
            for (int k = 0; k < 4; k++){
                vectorMeani[k] = meanRowi;
                vectorMeanj[k] = meanRowj;
            }
            if (nx % 4 == 0){
                conjugate_i = vectorMeani;
                conjugate_j = vectorMeanj;
            } else {
                for (int k = 0; k < nx % 4; k++){
                    conjugate_i[k] = meanRowi;
                    conjugate_j[k] = meanRowj;
                }
            }

            /*
            std::cout << "conjugate_i: ";
            for (int k = 0; k < 4; k++){
                std::cout << " " << conjugate_i[k] << " ";
            }
            */
            
            double4_t vectorNominator = double4_0;
            double4_t vectorSumi = double4_0;
            double4_t vectorSumj = double4_0;
            for(int k = 0; k < na - 1; k++){
                double4_t sumx = (vectorData[k + i*na] - vectorMeani);
                double4_t sumy = (vectorData[k + j*na] - vectorMeanj);
                vectorNominator += sumx * sumy;
                vectorSumi += sumx * sumx;
                vectorSumj += sumy * sumy;
            }

            double4_t sumx = (vectorData[na - 1 + i*na] - conjugate_i);
            double4_t sumy = (vectorData[na - 1 + j*na] - conjugate_j);
            vectorNominator += sumx * sumy;
            vectorSumi += sumx * sumx;
            vectorSumj += sumy * sumy;
            
            for (int k = 0; k < 4; k++){
                nominator += vectorNominator[k];
                sumi += vectorSumi[k];
                sumj += vectorSumj[k];
            }
            double denominator = sqrt(sumi * sumj);
            double cor = nominator / denominator;
            result[i + j * ny] = cor;
        }
    }   
    std::free(vectorData);
}

// .ppc/grader.py test
