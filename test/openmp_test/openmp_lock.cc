#include <time.h>
#include <omp.h>
#include <stdio.h>

void TestOmpLock()
{
    clock_t t1,t2;
    int i;
    int a = 0;

    omp_lock_t mylock;
    omp_init_lock(&mylock);
    t1 = clock();
    for( i = 0; i < 2000000; i++ )
    {
        omp_set_lock(&mylock);
        a+=1;
        omp_unset_lock(&mylock);
    }

    t2 = clock();
    printf("SingleThread,omp_lock 2,000,000:a = %ld, time = %ld\n", a, t2-t1);
    t1 = clock();

    #pragma omp parallel for
        for( i = 0; i < 2000000; i++ )
        {
        omp_set_lock(&mylock);
        a+=1;
        omp_unset_lock(&mylock);
        }

    t2 = clock();
    printf("MultiThread,omp_lock 2,000,000:a = %ld, time = %ld\n", a, t2-t1);
    omp_destroy_lock(&mylock);
}

int main(int argc, char* argv[])
{
    TestOmpLock();
    return 0;
}
