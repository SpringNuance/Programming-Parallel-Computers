#include <iostream>

using namespace std;

struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    int totalPixels = (x1 - x0) * (y1 - y0); 
    // std::cout << "ny: " << ny << ", nx: " << nx << ", x0: " << x0 << ", y0: " << y0 << ", x1: " << x1 << ", y1: " << y1 << ", totalPixels: " << totalPixels;
    double totalRed = 0.0;
    double totalGreen = 0.0; 
    double totalBlue = 0.0;
    for(int i = x0; i < x1; i++){
        for(int j = y0; j < y1; j++){
            // std::cout << "data: " << data[0 + 3 * i + 3 * nx * j];
            totalRed += data[0 + 3 * i + 3 * nx * j];
            totalGreen += data[1 + 3 * i + 3 * nx * j];
            totalBlue += data[2 + 3 * i + 3 * nx * j];
        }
    }
    // std::cout << "totalRed: " << totalRed << ", totalGreen: " << totalGreen << "totalBlue: " << totalBlue;    
    double averageRed = totalRed/totalPixels;
    double averageGreen = totalGreen/totalPixels;
    double averageBlue = totalBlue/totalPixels;
    Result result{{(float) averageRed, (float) averageGreen, (float)averageBlue}};
    return result;
}

// .ppc/grader.py test
