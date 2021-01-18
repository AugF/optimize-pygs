#include <iostream>
#include <stdio.h>
#include <stdlib.h>
// 必须的头文件
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

using namespace std;
 
#define NUM_THREADS 3
 
// 线程的运行函数
void *wait(void *t)
{
    int i;
    long id;
    
    id = (long)t;
    // https://blog.csdn.net/walilk/article/details/70432606
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("%ld begin time: %f\n", id, t1.tv_sec + t1.tv_nsec * 1.0 / 1000000);
    for (int i = 0, x = 1; i < 10000000; i ++) x += i;

    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("%ld end time: %f\n", id, t2.tv_sec + t2.tv_nsec * 1.0 / 1000000);
    // diff-second
    double time_diff_sec = (t2.tv_sec-t1.tv_sec) + (t2.tv_nsec-t1.tv_nsec) * 1.0 /1000000;
    printf("%ld use time: %f\n", id, time_diff_sec);
    pthread_exit(NULL);
}
 
int main()
{
   int rc;
   int i;
   pthread_t threads[NUM_THREADS];
   pthread_attr_t attr;
   void *status;
 
     struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
   // 初始化并设置线程为可连接的（joinable）
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   for( i=0; i < NUM_THREADS; i++ ){
      cout << "main() : creating thread, " << i << endl;
      rc = pthread_create(&threads[i], NULL, wait, (void *)&i );
      if (rc){
         cout << "Error:unable to create thread," << rc << endl;
         exit(-1);
      }
   }
 
   // 删除属性，并等待其他线程
   pthread_attr_destroy(&attr);
   for( i=0; i < NUM_THREADS; i++ ){
      rc = pthread_join(threads[i], &status);
      if (rc){
         cout << "Error:unable to join," << rc << endl;
         exit(-1);
      }
      cout << "Main: completed thread id :" << i ;
      cout << "  exiting with status :" << status << endl;
   }
 
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double time_diff_sec = (t2.tv_sec-t1.tv_sec) + (t2.tv_nsec-t1.tv_nsec) * 1.0 /1000000;
    printf("final use time: %f\n", time_diff_sec);
   cout << "Main: program exiting." << endl;
   pthread_exit(NULL);
}