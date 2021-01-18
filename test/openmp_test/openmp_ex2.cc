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


int main() {
   printf("cores:%d\n", omp_get_num_procs());
   printf("parent threadid:%d, cpu=%d\n",omp_get_thread_num(), sched_getcpu());
  
   int x = 0, num = 3;
   omp_lock_t mylock;
   omp_init_lock(&mylock);

   // x的第1位表示第1个数据是否产生，x的第0为表示第2个数据是否产生
   #pragma omp parallel sections num_threads(3)
   {
      #pragma omp section
      {
        for (int i = 0; i < num; i ++) {
            printf("begin fun1, i=%d, x=%d, time=%d\n", i, x, omp_get_thread_num());
            if ((x >> 1) & 1) printf("loop in fun1\n");
            while ((x >> 1) & 1);
            fun1(i);
            printf("fun1, i=%d, x=%d\n", i, x);
            omp_set_lock(&mylock);
            x |= 0b10;
            omp_unset_lock(&mylock);
            printf("after lock, fun1, i=%d, x=%d\n", i, x);
        }
      }

      #pragma omp section
      {  
         for (int i = 0; i < num; i ++) {
            printf("begin fun2, i=%d, x=%d, time=%d\n", i, x, omp_get_thread_num());
            if (!((x >> 1) & 1)) printf("loop in fun2 section1\n");
            while (!((x >> 1) & 1));
            omp_set_lock(&mylock);
            x &= ~0b10; // 修改状态
            omp_unset_lock(&mylock);

            fun2(i);
            printf("fun2, i=%d, x=%d, time=%d\n", i, x, omp_get_thread_num());
            if (x & 1) printf("loop in fun2 section2\n");// 当c正在被使用
            while (x & 1);
            omp_set_lock(&mylock);
            x |= 0b1; // 修改状态
            omp_unset_lock(&mylock);
            printf("after lock, fun2, i=%d, x=%d\n", i, x);
         }
      }

      #pragma omp section
      {
         for (int i = 0; i < num; i ++) {
            printf("begin fun3, i=%d, x=%d, time=%d\n", i, x, omp_get_thread_num());
            if (!(x & 1)) printf("loop in fun3\n");
            while (!(x & 1));
            fun3(i); 
            printf("fun3, i=%d, x=%d\n", i, x);
            omp_set_lock(&mylock);
            x &= ~0b1;
            omp_unset_lock(&mylock);
            printf("fun3, i=%d, x=%d\n", i, x);
         }
      }
   }
   omp_destroy_lock(&mylock);
   return 0;
}