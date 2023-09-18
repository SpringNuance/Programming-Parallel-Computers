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
#include "vector.h"

void correlate(int ny, int nx, const float *data, float *result) {
    // Calculate the mean of all rows
    std::vector<double> mean(ny);
    for(int i = 0; i < ny; i++){
        double sumRow = 0;
        for(int k = 0; k < nx; k++){
            sumRow += data[k + i*nx];
        }
        mean[i] = sumRow / nx;
    }

    int remainder = nx % 12;
    for(int i = 0; i < ny; i++){
        for(int j = 0; j <= i; j++){
            double meanRowi = mean[i];
            double meanRowj = mean[j];
            std::vector<double> sumi(12, 0.0); 
            std::vector<double> sumj(12, 0.0);
            std::vector<double> nom(12, 0.0);
            // correlation formula
            // r = sum((x_i - x_mean)(y_i - y_mean)) / sqrt(sum((x_i - x_mean)^2) * sum((y_i - y_mean)^2))
            for(int k = 0; k < nx/12; k++){
                for(int m = 0; m < 12; m++){
                    double sumx = (data[12 * k + m + i*nx] - meanRowi);
                    double sumy = (data[12 * k + m + j*nx] - meanRowj);
                    nom[m] += sumx * sumy;
                    sumi[m] += sumx * sumx;
                    sumj[m] += sumy * sumy;
                }
            }
            double nominator = 0.0;
            double sumI = 0.0;
            double sumJ = 0.0;
            for (int k = 0; k < 12; k++){
                nominator += nom[k];
                sumI += sumi[k];
                sumJ += sumj[k];
            }

            for(int k = 1; k <= remainder; k++){
                double sumx = (data[nx - k + i*nx] - meanRowi);
                double sumy = (data[nx - k + j*nx] - meanRowj);
                nominator += sumx * sumy;
                sumI += sumx * sumx;
                sumJ += sumy * sumy;
            }
            
            /*
            for(int k = 0; k < nx; k++){
                sumx = (data[k + i*nx] - meanRowi);
                sumy = (data[k + j*nx] - meanRowj);
                nominator += sumx * sumy;
                sumi += sumx * sumx;
                sumj += sumy * sumy;
            }
            */
            double denominator = sqrt(sumI * sumJ);
            
            double cor = nominator / denominator;
            result[i + j * ny] = cor;
        }
    }   
}

// .ppc/grader.py test