#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <sched.h>
#include <unistd.h>


void fun1(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 5000000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun1: begin_time %13.8f, end_time %13.8f\n", id, st1, st2);
   // printf("begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

void fun2(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 5000000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun2: begin_time %13.8f, end_time %13.8f\n", id, st1, st2);
   // printf("begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

void fun3(int id) {
   int a = 0;
   double st1 = omp_get_wtime();
   for (int i = 0; i < 5000000; i ++) {
      a = i + 1;
   }
   double st2 = omp_get_wtime();
   printf("%d_fun3: begin_time %13.8f, end_time %13.8f\n", id, st1, st2);
   // printf("begin time: %13.8f, end time: %13.8f, Time = %13.8f, a=%d, Thread_id=%d, cpu=%d\n", st1, st2, st2 - st1, a, omp_get_thread_num(), sched_getcpu());
}

int main() // argc为参数总数，argv为参数数组
{
   int num = 5;

   double t1 = omp_get_wtime();
   int x = 0;
   omp_lock_t mylock;
   omp_init_lock(&mylock);

   // x的第1位表示第1个数据是否产生，x的第0为表示第2个数据是否产生
   #pragma omp parallel sections num_threads(3) 
   {
      #pragma omp section
      {
        for (int i = 0; i < num; i ++) {
            while ((x >> 1) & 1); 
            fun1(i);
            omp_set_lock(&mylock);
            x |= 0b10;
            omp_unset_lock(&mylock);
        }
      }

      #pragma omp section
      {  
         for (int i = 0; i < num; i ++) {
            while (!((x >> 1) & 1));
            
            omp_set_lock(&mylock);
            x &= ~0b10; // 修改状态
            omp_unset_lock(&mylock);

            fun2(i);
            while (x & 1); // 当c正在被使用
            omp_set_lock(&mylock);
            x |= 0b1; // 修改状态
            omp_unset_lock(&mylock);
         }
      }

      #pragma omp section
      {
         int b, c;
         for (int i = 0; i < num; i ++) {
            while (!(x & 1));
            fun3(i); // 读取
            omp_set_lock(&mylock);
            x &= ~0b1;
            omp_unset_lock(&mylock);
         }
      }
   }
   omp_destroy_lock(&mylock);
   double t2 = omp_get_wtime();
   printf("total use time: %13.8fs\n", t2 - t1);

   return 0;
}
