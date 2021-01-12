#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>
#include <unistd.h>
int main()
{

   printf("parent threadid:%d\n",omp_get_thread_num());
   clock_t t1 = clock();
   #pragma omp sections
   {
     #pragma omp section
     {
          printf("section 0,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
     #pragma omp section
     {
          printf("section 1,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
     #pragma omp section
     {
          printf("section 2,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
   }

   clock_t t2 = clock();
   #pragma omp parallel sections
   {
      #pragma omp section
     {
          printf("section 3,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
      #pragma omp section
     {
          printf("section 4,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
      #pragma omp section
     {
          printf("section 5,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
   }
   
   clock_t t3 = clock();
   printf("CLOCKS_PER_SEC=%d\n", CLOCKS_PER_SEC);
   printf("t1=%d, t2=%d, t3=%d\n", t1, t2, t3);
   printf("sections use time: %d, parallel sections use time: %d\n", t2 - t1, t3 - t2);
 
 return 0;
}
