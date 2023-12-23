#include <iostream>
#include <math.h>
#include <vector>  
#include <iostream>
#include "vector.h"
#include <chrono>
#include <algorithm>
#include <omp.h>

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
// inclusion-exclusion helper function to compute the sum of the inner rectangle in O(1) time
double4_t sumInner(int x0, int y0, int sx, int sy, int nxCorners, std::vector<double4_t>& sumCorners) {
    int x1 = x0 + sx;
    int y1 = y0 + sy; 
    double4_t sumBigRec = sumCorners[x1 + nxCorners * y1];
    double4_t sumSmallRec = sumCorners[x0 + nxCorners * y0];
    double4_t sumUpperRec = sumCorners[x1 + nxCorners * y0];
    double4_t sumLeftRec = sumCorners[x0 + nxCorners * y1];
    return sumBigRec + sumSmallRec - sumUpperRec - sumLeftRec;
}

Result minResult(int& ny, int& nx, double4_t& sumGrid, std::vector<double4_t>& sumCorners, std::vector<double>& minErrorsSize, std::vector<std::pair<int,int>>& minx0y0Size){
    int y0min = 0;
    int x0min = 0; 
    int y1min = 0; 
    int x1min = 0;
    double4_t outerMin = double4_0;
    double4_t innerMin = double4_0;
    double minErrorsCurrent = std::numeric_limits<double>::infinity();
    //Result result{0, 0, 0, 0, {0,0,0}, {0,0,0}};
    for (int sy = 0; sy < ny; sy++){
        for (int sx = 0; sx < nx; sx++){
            if (minErrorsCurrent > minErrorsSize[sx + nx * sy]){
                minErrorsCurrent = minErrorsSize[sx + nx * sy];
                y0min = minx0y0Size[sx + nx * sy].second; 
                x0min = minx0y0Size[sx + nx * sy].first;
                y1min = minx0y0Size[sx + nx * sy].second + sy + 1; 
                x1min = minx0y0Size[sx + nx * sy].first + sx + 1;
            }
        }
    }
    int width = y1min - y0min;
    int length = x1min - x0min;
    double areaInner = width * length;
    double areaOuter = nx * ny - areaInner;
    int nxCorners = nx + 1;
    double4_t innerSum = sumInner(x0min, y0min, length, width, nxCorners, sumCorners);
    double4_t outerSum = sumGrid - innerSum;
    innerMin = innerSum/areaInner; // y_inner
    outerMin = outerSum/areaOuter; // y_outer
    Result result{y0min, x0min, y1min, x1min, {(float)outerMin[0], (float)outerMin[1], (float)outerMin[2]}, {(float)innerMin[0], (float)innerMin[1], (float)innerMin[2]}};    
    return result;
}

Result segment(int ny, int nx, const float *data) {
    // vectorData, each vector contains 3 values of RGB and a padded 0 in double4_t
    double4_t* vectorData = double4_alloc(ny*nx);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                double num = data[c + 3 * x + 3 * nx * y];
                vectorData[x + nx * y][c] = num;
            }
            vectorData[x + nx * y][3] = 0.0;
        }
    }

    // Sum of corner rectangles, which will be used for inclusion-exclusion principles
    int nyCorners = ny + 1;
    int nxCorners = nx + 1;
    std::vector<double4_t> sumCorners(nyCorners*nxCorners, double4_0);
    //std::vector<double4_t> sumCornersSquared(nyCorners*nxCorners, double4_0);
    
    #pragma omp parallel for
    for (int sy = 1; sy < nyCorners; sy++){
        double4_t sum = double4_0;
        for (int ky = 1; ky <= sy; ky++){
            sum += vectorData[(ky - 1) * nx];
        } 
        sumCorners[nxCorners * sy + 1] = sum;
    }
    
    #pragma omp parallel for
    for (int sx = 1; sx < nxCorners; sx++){
        double4_t sum = double4_0;
        for (int kx = 1; kx <= sx; kx++){
            sum += vectorData[kx - 1];
        } 
        sumCorners[sx + nxCorners] = sum;
    }

    // cannot be parallelized with openMP because this is a dependent read and write. Although dependent, it is faster than the normal parallel version
    // Calculating the sumCorners and sumCornersSquared. This method is faster than parallelization because it uses memoization
    
    for (int sy = 2; sy < nyCorners; sy++){
        for (int sx = 2; sx < nxCorners; sx++){
            sumCorners[sx + nxCorners * sy] = vectorData[(sx - 1) + nx * (sy - 1)] + sumCorners[(sx - 1) + nxCorners * sy] + sumCorners[sx + nxCorners * (sy - 1)] - sumCorners[(sx - 1) + nxCorners * (sy - 1)];
        }
    }

    // sx, sy tries out all possible sizes of the rectangle
    double4_t sumGrid = sumInner(0, 0, nx, ny, nxCorners, sumCorners);
    double4_t sumGridSquared = sumGrid * sumGrid;
    std::vector<double> minErrorsSize(ny * nx);
    std::vector<std::pair<int,int>> minx0y0Size(ny * nx);
    auto start = std::chrono::high_resolution_clock::now();
    double area = nx*ny;
    #pragma omp parallel for schedule(dynamic, 1) 
    for (int sy = 0; sy < ny; sy++){
        for (int sx = 0; sx < nx; sx++){
            int width = sy + 1;
            int length = sx + 1;
            double areaInner = width * length;
            double areaOuter = nx * ny - areaInner;
            double alpha = 1/areaInner;
            double beta = 1/areaOuter;
            double gamma = alpha + beta;
            double4_t lambda = 2 * beta * sumGrid;
            //double division = 1/(areaInner * (nx * ny - areaInner));
            //double4_t sumGridAreaInner = sumGrid * areaInner;
            double minErrorsCurrent = std::numeric_limits<double>::infinity();
            for (int y0 = 0; y0 < ny - sy; y0++){
                for (int x0 = 0; x0 < nx - sx; x0++){
                    double4_t innerSum = sumInner(x0, y0, length, width, nxCorners, sumCorners);
                    //double4_t tempError = (2.0 * sumGrid * innerSum * areaInner - innerSum * innerSum * area - sumGridSquared * areaInner) * division;
                    //double4_t tempError = (innerSum * (sumGrid * areaInner - innerSum * area) + sumGridAreaInner * (innerSum - sumGrid)) * division;
                    double4_t outerSum = sumGrid - innerSum;
                    //double4_t tempError = -((innerSum * innerSum)/areaInner + (outerSum * outerSum)/areaOuter);
                    double4_t tempError = -((innerSum * innerSum)/areaInner + (outerSum * outerSum)/areaOuter);
                    double totalErrors = tempError[0] + tempError[1] + tempError[2];
                    if (minErrorsCurrent > totalErrors){
                        minErrorsCurrent = totalErrors;
                        minx0y0Size[sx + nx * sy] = std::make_pair(x0, y0);
                    }
                }
            }
            minErrorsSize[sx + nx * sy] = minErrorsCurrent;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    Result finalResult = minResult(ny, nx, sumGrid, sumCorners, minErrorsSize, minx0y0Size);
    std::free(vectorData);
    std::cout << "This code part takes " << (end - start).count() << "s\n";
    return finalResult;
}

// .ppc/grader.py test