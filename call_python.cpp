#include <stdio.h>
#include <pybind11/pybind11.h>
#include "funs.cpp"
using namespace pybind11::literals;

namespace py = pybind11;

int run() {
    py::module_ tasks = py::module_::import("code.optimize_batch.tasks");
    // py::module_ time = py::module_::import("time");

    // attr
    py::object model = tasks.attr("model");
    py::object optimizer = tasks.attr("optimizer");
    py::object device = tasks.attr("device");
    py::object loader = tasks.attr("loader");
    py::object loader_iter1 = py::iter(loader);
    py::object loader_iter2 = py::iter(loader);
    
    // res
    float total_loss = 0.0;
    long long total_acc = 0, total_num = 0;

    // funs
    py::object task1 = tasks.attr("task1");
    py::object task2 = tasks.attr("task2");
    py::object task3 = tasks.attr("task3");

    // py::object t1 = time.time();
    // single
    for (int i = 0; i < int(py::len(loader)); i ++) {
        // do tasks
        py::object data_cpu = task1(loader_iter1);
        py::object data_gpu = task2(data_cpu, device);
        py::list res = task3(data_gpu, model, optimizer);

        // update parameters
        model = res[0];
        optimizer = res[1];
        // 类型转换
        total_loss += res[2].cast<float>(); 
        total_acc += res[3].cast<long long>();
        total_num += res[4].cast<long long>(); 
        // print res
        std::printf("run%2d: total_loss: %.4f, total_acc: %lld, total_num: %lld\n", i, total_loss, total_acc, total_num);
    }

    // parallel
    for (int i = 0; i < int(py::len(loader)); i ++) {
        // do tasks
        py::object data_cpu = task1(loader_iter2);
        py::object data_gpu = task2(data_cpu, device);
        py::list res = task3(data_gpu, model, optimizer);

        // update parameters
        model = res[0];
        optimizer = res[1];
        // 类型转换
        total_loss += res[2].cast<float>(); 
        total_acc += res[3].cast<long long>();
        total_num += res[4].cast<long long>(); 
        // print res
        std::printf("run%2d: total_loss: %.4f, total_acc: %lld, total_num: %lld\n", i, total_loss, total_acc, total_num);
    }

    // py::print("use time", t2 - t1);
    return 0;
}


int run_parallel() {
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
   
   return 0;
}

PYBIND11_MODULE(call_python, m) {
    m.doc() = "pybind11 call_python plugin";
    m.def("run", &run);
    m.def("run_parallel", &run_parallel);
}