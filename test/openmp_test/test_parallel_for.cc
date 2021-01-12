#include <iostream>
#include <cstdio>
#include <time.h>
#include <omp.h>

using namespace std;

int main() {
    /*#pragma omp parallel num_threads(8)
    {
        printf("Hello World!, ThreadId=%d\n", omp_get_thread_num());
    }

    #pragma omp parallel for num_threads(2)
    //{
        for (int j = 0; j < 4; j ++) {
            printf("j=%d, ThreadId=%d\n", j, omp_get_thread_num());
        }
    //}
    */
    #pragma omp sections {
    #pragma omp section
        printf("section 1 ThreadId=%d\n", omp_get_thread_num());
    #pragma omp section
        printf("section 2 ThreadId=%d\n", omp_get_thread_num());
    #pragma omp section
        printf("section 3 ThreadId=%d\n", omp_get_thread_num());
    }

    //return 0;
}
