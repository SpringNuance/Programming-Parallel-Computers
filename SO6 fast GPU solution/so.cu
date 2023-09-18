#include <algorithm>

typedef unsigned long long data_t;

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the GPU.
    std::sort(data, data + n);
}
