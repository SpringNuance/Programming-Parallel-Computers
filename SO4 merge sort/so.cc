#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>

typedef unsigned long long data_t;


void mergeHelper(data_t *data, int left, int mid, int right){
    std::inplace_merge(data + left, data + mid + 1, data + right + 1);
}

void printArray(int n, data_t* data, int p){
    std::cout << "Sorted array at number of threads " << p << std::endl;
    for (int i = 0; i < n; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void mergeSort(data_t *data, int left, int right, int level) {
  if (left >= right || level == 0) {
    // m is the point where the array is divided into two subarrays
    std::sort(data + left, data + right + 1);
  } else {
    // Merge the sorted subarrays
    int portion = (right - left) / 4;
    int mid1 = left + portion;
    int mid2 = left + 2 * portion;
    int mid3 = left + 3 * portion;
    #pragma omp task
    mergeSort(data, left, mid1, level - 1);

    #pragma omp task
    mergeSort(data, mid1 + 1, mid2, level - 1);

    #pragma omp task
    mergeSort(data, mid2 + 1, mid3, level - 1);

    #pragma omp task
    mergeSort(data, mid3 + 1, right, level - 1);
    
    #pragma omp taskwait
    
    #pragma omp task
    mergeHelper(data, left, mid1, mid2);

    #pragma omp task
    mergeHelper(data, mid2 + 1, mid3, right);

    #pragma omp taskwait

    mergeHelper(data, left, mid2, right);

  }
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    int level = 3;
    // std::sort(data, data + n);
    #pragma omp parallel
    #pragma omp single
    {
      mergeSort(data, 0, n - 1, level);
    }

}

// .ppc/grader.py test


