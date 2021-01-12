#include <iostream>
#include <cstdio>
#include <time.h>
#include <omp.h>
using namespace std;

void test() {
   int a = 0;
   clock_t t1 = clock();
   for (int i = 0; i < 100000000; i ++) {
      a = i + 1;
   }
   clock_t t2 = clock();
   printf("Time = %d\n", t2 - t1);
}

int main() {

   printf("%d\n", CLOCKS_PER_SEC);

   clock_t t1 = clock();
#pragma omp parallel for
   for (int j = 0; j < 2; j ++) {
      test();
   }
   clock_t t2 = clock();
   printf("Total time = %d\n", t2 - t1);

   test();
   return 0;
}
