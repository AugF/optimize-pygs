#include <stdio.h>
#include <omp.h>
#include <time.h>

void test(int k) {
    printf("test k=%d\n", k);
}


int main() {
    int k = 100;
    
    #pragma omp parallel for firstprivate(k),lastprivate(k)
        for (int i = 0; i < 41; i ++) {
            k += i;
            test(k);
            //printf("procs=%d, threads=%d\n", omp_get_num_procs(), omp_get_num_threads());
            printf("k = %d, threadid=%d\n", k, omp_get_thread_num());
        }

    printf("last k=%d\n", k);
}
