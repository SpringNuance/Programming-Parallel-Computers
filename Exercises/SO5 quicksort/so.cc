#include <algorithm>
#include <iostream>     
#include <algorithm>   
#include <vector>      

typedef unsigned long long data_t;
 
void quickSort(data_t* left, data_t * right, int level){
 
    // base case
    if (left >= right || level == 0){
        std::sort(left, right);
    } else {
    // 3-way quicksort partitioning the array
    auto pivot = *(left + (rand() % (right - left)));
    auto middle2 = std::partition(left, right, [pivot](data_t& num){return num <= pivot;});
    auto middle1 = std::partition(left, middle2, [pivot](data_t& num){return num < pivot;});
 
    // Sorting the left part
    #pragma omp task
    quickSort(left, middle1, level - 1);
    #pragma omp task
    // Sorting the right part
    quickSort(middle2, right, level - 1);
    }
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    int level = 10;
    #pragma omp parallel
    #pragma omp single
    {
    quickSort(data, data + n, level);
    }
}

// .ppc/grader.py test
