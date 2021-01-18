#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sched.h>
#include <unistd.h>

void fun1(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 50000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun1 begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", id, st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

void fun2(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 50000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun2 begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", id, st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

void fun3(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 50000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun3 begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", id, st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

void test() {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 1000000000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}