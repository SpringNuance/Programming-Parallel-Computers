// CPP program for implementation of QuickSelect
#include <bits/stdc++.h>
using namespace std;
#include <vector>
#include <cassert>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/


void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  int loweri, upperi, lowerj, upperj, numPixels, median;
  std::vector<double> array;
  for (int x = 0; x < nx; x++){
    for (int y = 0; y < ny; y++){
      // Finding the range of the sliding window
      if (x - hx < 0){
        loweri = 0;
      } else loweri = x - hx;
      if (y - hy < 0){
        lowerj = 0;
      } else lowerj = y - hy;
      if (x + hx > nx - 1){
        upperi = nx - 1;
      } else upperi = x + hx;
      if (y + hy >= ny - 1){
        upperj = ny - 1;
      } else upperj = y + hy;
      //std::cout << "loweri: " << loweri << ", upperi: " << upperi << ", lowerj: " << lowerj << ", upperj: " << upperj << std::endl;
      // Adding all pixel values of the sliding window into the an array
      numPixels = (upperi - loweri + 1) * (upperj - lowerj + 1);
      //std::cout << "numPixels: " << numPixels << std::endl;
      int k = 0;
      for (int i = loweri; i <= upperi; i++){
        for (int j = lowerj; j <= upperj; j++){
          array.push_back(in[i + j * nx]);
          k += 1;
        }
      }
      //std::nth_element(array.begin(), array.begin() + 1, array.end(), std::less);
      if (numPixels % 2 == 1){
        std::nth_element(array.begin(), array.begin() + numPixels/2, array.end());
        out[x + y * nx] = (double) array[numPixels/2];
      } else {
        std::nth_element(array.begin(), array.begin() + numPixels/2, array.end());
        out[x + y * nx] +=  (double)(array[numPixels / 2]);
        std::nth_element(array.begin(), array.begin() + (numPixels - 1)/2, array.end());
        out[x + y * nx] +=  (double)(array[(numPixels - 1)/2]);
        out[x + y * nx] /= 2.0;
      }     
      array.clear();
    }
  }
}

// .ppc/grader.py test