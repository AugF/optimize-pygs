#include <iostream>
#include <omp.h>
void test()
{
    double t1 = omp_get_wtime();
    int a = 0;
    for (int i=0;i<100000000;i++)
        a++;
    std::cout << omp_get_thread_num() << ", use time: " << omp_get_wtime() - t1 << std::endl;
}
int main()
{
    omp_set_num_threads(omp_get_max_threads());
    double t1 = omp_get_wtime();
    #pragma omp parallel for num_threads(8)
    for (int i=0;i<8;i++)
        test();
    double t2 = omp_get_wtime();
    std::cout<<"time: "<<t2-t1<<std::endl;

    t1 = omp_get_wtime();
    for (int i = 0; i < 8; i ++) test();
    t2 = omp_get_wtime();
    std::cout << "time: " << t2 - t1 << std::endl;
    return 0;
}