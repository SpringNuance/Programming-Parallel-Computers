#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>

typedef unsigned long long data_t;

void mergeHelper(data_t *data, int left, int mid, int right){
    int leftSize = mid - left;
    int rightSize = right - mid;
  
    // Create temp arrays
    std::vector<data_t> leftTemp(leftSize);
    std::vector<data_t> rightTemp(rightSize);
  
    // Copy data to temp arrays
    for (int i = 0; i < leftSize; i++)
        leftTemp[i] = data[left + i];
    for (int j = 0; j < rightSize; j++)
        rightTemp[j] = data[mid + j];
    std::merge(leftTemp.begin(), leftTemp.end(), rightTemp.begin(), rightTemp.end(), data + left);
}

void mergeSort(data_t *data, int left, int right, int baseBlockSize) {
  if (right - left == baseBlockSize) {
    std::sort(data + left, data + right);
  } else {
    int mid = left + (right - left) / 2;
    mergeHelper(data, left, mid, right);
  }
}

void printArray(int n, data_t* data, int p){
    std::cout << "Sorted array at number of threads " << p << std::endl;
    for (int i = 0; i < n; i++){
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    // Divide and conquer — bottom up
    // std::sort(data + 0, data + n);
    
    int p = omp_get_max_threads();
    //std::cout << "Number of threads is " << p << std::endl;
    int baseblockSize = n / p;
    int blockSize = n / p;
    int remainder = n % p;
    int indexMerge = n - remainder;
    std::sort(data + indexMerge, data + n);
    //printArray(n, data, p);
    while (p > 1) {
        //std::cout << "Block size is " << blockSize << std::endl;
        //std::cout << "indexMerge is " << indexMerge << std::endl;
        
        #pragma omp parallel num_threads(p)   
        {
        int i = omp_get_thread_num();     
        mergeSort(data, i *  blockSize, (i + 1) * blockSize, baseblockSize);
        }
        
        #pragma omp taskwait
        
        //printArray(n, data, p);
        if (p % 2 == 1) {
            mergeHelper(data, (p - 1) * blockSize, indexMerge, n);
            indexMerge -= blockSize;
        }
        p = p/2;
        blockSize *= 2;
        
    }
    //std::cout << "Block size is " << blockSize << std::endl;
    //std::cout << "indexMerge is " << indexMerge << std::endl;
    mergeSort(data, 0, blockSize, baseblockSize);
    mergeHelper(data, 0, indexMerge, n);
    //printArray(n, data, p);
    
    mergeHelper(data, 0, n/2, n);
}

// .ppc/grader.py test
