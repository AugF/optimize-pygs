#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sched.h>
#include <unistd.h>

void test() {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 1000000000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}
// why test time is different, 230000, 680000

int main()
{
   printf("cores:%d\n", omp_get_num_procs());
   printf("parent threadid:%d, cpu=%d\n",omp_get_thread_num(), sched_getcpu());
   double t1 = omp_get_wtime();
   // #pragma omp sections 
   // {
   //   #pragma omp section
   //   {
   //      //   printf("section 0,threadid=%d\n",omp_get_thread_num());
   //        test();
   //   }
   //   #pragma omp section
   //   {
   //      //   printf("section 1,threadid=%d\n",omp_get_thread_num());
   //        test();
   //   }
   //   #pragma omp section
   //   {
   //      //   printf("section 2,threadid=%d\n",omp_get_thread_num());
   //        test();
   //   }
   // }

   double t2 = omp_get_wtime();
   #pragma omp parallel sections num_threads(3)
   {
      #pragma omp section
     {
        //   printf("section 3,threadid=%d\n",omp_get_thread_num());
          test();
     }
      #pragma omp section
     {
        //   printf("section 4,threadid=%d\n",omp_get_thread_num());
          test();
     }
      #pragma omp section
     {
        //   printf("section 5,threadid=%d\n",omp_get_thread_num());
          test();
     }
   }
   
   double t3 = omp_get_wtime();
   printf("t1=%13.8f, t2=%13.8f, t3=%13.8f\n", t1, t2, t3);
   printf("sections use time: %13.8f, parallel sections use time: %13.8f\n", t2 - t1, t3 - t2);
   
   return 0;
}
