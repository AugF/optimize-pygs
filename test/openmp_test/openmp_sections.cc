#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include <unistd.h>
int main()
{

   printf("parent threadid:%d\n",omp_get_thread_num());
   #pragma omp  sections
   {
     #pragma omp section
     {
          printf("section 0,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
     #pragma omp section
     {
          printf("section 1,threadid=%d\n",omp_get_thread_num());
          //sleep(1);
     }
     #pragma omp section
     {
          printf("section 2,threadid=%d\n",omp_get_thread_num());
          sleep(1);
     }
   }
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
 
 return 0;
}