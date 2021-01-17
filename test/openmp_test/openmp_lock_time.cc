#include <unistd.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int main()
{
    clock_t clock_timer;
    double wall_timer;
    double c[10000];
    for (int nthreads = 1; nthreads <= omp_get_max_threads() - 1; ++nthreads)
    {
            clock_timer = clock();
            wall_timer = omp_get_wtime();
            printf("threads: %d ", nthreads);
            #pragma omp parallel num_threads(nthreads)
            for (int i = 0; i < 10000; i++)
            {
                    for (int j = 0; j < 10000; j++)
                            c[i] = sqrt(i * 4 + j * 2 + j);
            }
            printf("time on clock(): %13.8f s\t", (double) (clock() - clock_timer) / CLOCKS_PER_SEC);
            printf("time on wall: %13.8f s\n", omp_get_wtime()-wall_timer);
    }
    
    return 0;
}