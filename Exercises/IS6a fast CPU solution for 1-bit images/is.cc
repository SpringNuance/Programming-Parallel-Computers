#include <iostream>
#include <math.h>
#include <vector>  
#include <iostream>
#include "vector.h"
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <x86intrin.h>
#include <immintrin.h>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

constexpr float infty = std::numeric_limits<float>::infinity();

constexpr float8_t f8infty {
    infty, infty, infty, infty,
    infty, infty, infty, infty
};

static inline float hmin8(float8_t vv) {
    float v = infty;
    for (int i = 0; i < 8; ++i) {
        if (vv[i] < v){
            v = vv[i];
        }
    }
    return v;
}

// inclusion-exclusion helper function to compute the sum of the inner rectangle in O(1) time
// Calculating the sum of only 1 rectangle
float sumInnerFloat(int x0, int y0, int sx, int sy, int& nxCorners, std::vector<float>& sumCornersNormal) {
    int x1 = x0 + sx;
    int y1 = y0 + sy; 
    float sumBigRec = sumCornersNormal[x1 + nxCorners * y1];
    float sumSmallRec = sumCornersNormal[x0 + nxCorners * y0];
    float sumUpperRec = sumCornersNormal[x1 + nxCorners * y0];
    float sumLeftRec = sumCornersNormal[x0 + nxCorners * y1];
    return sumBigRec + sumSmallRec - sumUpperRec - sumLeftRec;
}

static inline float8_t min8(float8_t x, float8_t y) {
    return x < y ? x : y;
}

// inclusion-exclusion helper function to compute the sum of the inner rectangle in O(1) timeS
// Using a vector of 8, calculating innerSum of 8 rectangles of size sx x sy with moving window along x-axis in one go.
float8_t sumInnerFloatVector(int x0, int y0, int sx, int sy, int& nxCorners, std::vector<float>& sumCornersNormal) {
    int x1 = x0 + sx;
    int y1 = y0 + sy; 
    // "unaligned load" instructions
    float8_t sumBigRec = _mm256_loadu_ps(&sumCornersNormal[x1 + nxCorners * y1]);
    float8_t sumSmallRec = _mm256_loadu_ps(&sumCornersNormal[x0 + nxCorners * y0]);
    float8_t sumUpperRec = _mm256_loadu_ps(&sumCornersNormal[x1 + nxCorners * y0]);
    float8_t sumLeftRec = _mm256_loadu_ps(&sumCornersNormal[x0 + nxCorners * y1]);
    return sumBigRec + sumSmallRec - sumUpperRec - sumLeftRec;
}

Result minResultFloat(int& ny, int& nx, float& sumGrid, std::vector<float>& sumCornersNormal, std::vector<float>& minErrorsSize){
    int y0min = 0;
    int x0min = 0; 
    int y1min = 0; 
    int x1min = 0;
    float outerMin = 0;
    float innerMin = 0;
    int index = std::distance(std::begin(minErrorsSize), std::min_element(std::begin(minErrorsSize), std::end(minErrorsSize)));
    int sxMin = index % nx;
    int syMin = index / nx;
    int width = syMin + 1;
    int length = sxMin + 1;
    float areaInner = width * length;
    float areaOuter = nx * ny - areaInner;
    
    float dividedbyareaInner = 1/areaInner;
    float dividedbyareaOuter = 1/areaOuter;
    float alpha = dividedbyareaInner + dividedbyareaOuter;
    float beta = 2 * sumGrid * dividedbyareaOuter;
    int nxCorners = nx + 1;
    float sumGridSquared = sumGrid * sumGrid;
    float minErrorsCurrent = std::numeric_limits<float>::infinity();
    //float minErrorsCurrent = std::numeric_limits<float>::infinity();
    for (int y0 = 0; y0 < ny - syMin; y0++){
        for (int x0 = 0; x0 < nx - sxMin; x0 ++){
            float innerSum = sumInnerFloat(x0, y0, length, width, nxCorners, sumCornersNormal);
            float totalErrors = -(innerSum * innerSum * alpha + sumGridSquared * dividedbyareaOuter - beta*innerSum);
            if (minErrorsCurrent > totalErrors){
                minErrorsCurrent = totalErrors;
                y0min = y0;
                x0min = x0;
            }
        }
    }
    y1min = y0min + width;
    x1min = x0min + length;
    float innerSum = sumInnerFloat(x0min, y0min, length, width, nxCorners, sumCornersNormal);
    float outerSum = sumGrid - innerSum;
    innerMin = innerSum/areaInner; // y_inner
    outerMin = outerSum/areaOuter; // y_outer
    Result result{y0min, x0min, y1min, x1min, {outerMin, outerMin, outerMin}, {innerMin, innerMin, innerMin}};    
    return result;
}

Result segment(int ny, int nx, const float *data) {
    // vectorData, each vector contains 3 values of RGB and a padded 0 in double4_t
    //auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> dataNormal(ny*nx);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            // 1 bit image so extracting the first pixel is enough to know whether it is black or white
            dataNormal[x + nx * y] = data[3 * x + 3 * nx * y]; 
        }
    }
    
    int nyCorners = ny + 1;
    int nxCorners = nx + 1;

    std::vector<float> sumCornersNormal(nyCorners*nxCorners, 0.0);
    #pragma omp parallel for
    for (int sy = 1; sy < nyCorners; sy++){
        double sum = 0;
        for (int ky = 1; ky <= sy; ky++){
            sum += dataNormal[(ky - 1) * nx];
        } 
        sumCornersNormal[nxCorners * sy + 1] = sum;
    }
    #pragma omp parallel for
    for (int sx = 1; sx < nxCorners; sx++){
        double sum = 0;
        for (int kx = 1; kx <= sx; kx++){
            sum += dataNormal[kx - 1];
        } 
        sumCornersNormal[sx + nxCorners] = sum;
    }
    for (int sy = 2; sy < nyCorners; sy++){
        for (int sx = 2; sx < nxCorners; sx++){
            sumCornersNormal[sx + nxCorners * sy] = dataNormal[(sx - 1) + nx * (sy - 1)] + sumCornersNormal[(sx - 1) + nxCorners * sy] + sumCornersNormal[sx + nxCorners * (sy - 1)] - sumCornersNormal[(sx - 1) + nxCorners * (sy - 1)];
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    float sumGrid = sumInnerFloat(0, 0, nx, ny, nxCorners, sumCornersNormal);
    float sumGridSquared = sumGrid * sumGrid;
    std::vector<float> minErrorsSize(ny * nx);
    std::vector<std::pair<int,int>> minx0y0Size(ny * nx);
    #pragma omp parallel for schedule(dynamic, 1) 
    for (int sy = 0; sy < ny; sy++){
        for (int sx = 0; sx < nx; sx++){
            float areaInner = (sy + 1) * (sx + 1);
            float8_t minErrors = f8infty;
            int na = (nx - sx)/8;
            float dividedbyareaInner = 1/areaInner;
            float dividedbyareaOuter = 1/(nx*ny-areaInner);
            float alpha = dividedbyareaInner + dividedbyareaOuter;
            float beta = 2 * sumGrid * dividedbyareaOuter;
            float gamma = sumGridSquared * dividedbyareaOuter;
            for (int y0 = 0; y0 < ny - sy; y0++){
                for (int x0 = 0; x0 < na * 8; x0 += 8){
                    float8_t innerSum = sumInnerFloatVector(x0, y0, sx + 1, sy + 1, nxCorners, sumCornersNormal);
                    float8_t totalErrors = (beta - innerSum * alpha) * innerSum - gamma;
                    minErrors = min8(minErrors,totalErrors);
                }
            }
            float minErrorsCurrent = hmin8(minErrors);

            for (int y0 = 0; y0 < ny - sy; y0++){
                for (int x0 = na * 8; x0 < nx - sx; x0 ++){
                    float innerSum = sumInnerFloat(x0, y0, sx + 1, sy + 1, nxCorners, sumCornersNormal);
                    float totalErrors = (beta - innerSum * alpha) * innerSum - gamma;
                    if (minErrorsCurrent > totalErrors){
                        minErrorsCurrent = totalErrors;
                    }
                }
            }
            minErrorsSize[sx + nx * sy] = minErrorsCurrent;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    Result finalResult = minResultFloat(ny, nx, sumGrid, sumCornersNormal, minErrorsSize);
    //std::cout << "This code part takes " << (end - start).count() << "s\n";
    return finalResult;
}

// .ppc/grader.py test

// ./grading test-plain tests/020-small-structured.txt