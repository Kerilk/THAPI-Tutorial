---
title: 'Tracing Heterogeneous APIs (OpenCL, L0, CUDA) in a nutshell: billions of events with low overhead'
author:
- Brice Videau
- Thomas Applencourt
institute:
- Argonne National Laboratory
date: 14th April 2021
theme: Warsaw
header-includes:
- |
  ```{=latex}
  \usepackage{./style/beamercolorthemeprogressbar}
  \usepackage{./style/beamerfontthemeprogressbar}
  \usepackage{./style/beamerouterthemeprogressbar}
  \usepackage{./style/beamerinnerthemeprogressbar}
  \setbeamertemplate{navigation symbols}{}
  \beamertemplatetransparentcovereddynamic
  \usepackage{pdfpcnotes}
  ```
---

# Context

## Intro

Programming languages and models for HPC have never been more diverse:

:::::::::::::: {.columns}
::: {.column width="35%"}
### Languages

 * FORTRAN
 * C
 * C++
 * Python

### Prospective languages

 * Julia
 * Lua
 * Chapel
 * PGAS approaches

:::
::: {.column width="65%"}
### Programming models

 * MPI
 * OpenMP
 * CUDA, L0, ROCm, HIP, OpenCL
 * SYCL/DPC++
 * Kokkos
 * Raja

### Domain Based Programming Models

 * Linear algebra: BLAS/LAPACK
 * FFTs: cuFFT, FFTWx, mkl FFT
 * Low level AI: cuDNN, clDNN, Intel DNNL
 * AI/ML: TensorFlow/Caffe/PyTorch

:::
::::::::::::::

## Problematic

This plethora of alternatives are entwined, especially since
heterogeneous computing is the norm.

:::::::::::::: {.columns}
::: {.column width="55%"}
### Possible Dependencies
:::::::::::::: {.columns}
::: {.column width="45%"}

 * SYCL:
   - HIP
   - OpenCL
   - L0
 * OpenMP:
   - OpenCL
   - CUDA
   - L0
 * OpenCL:
   - L0
   - CUDA
:::
::: {.column width="50%"}
 * HIP:
   - CUDA
   - OpenCL
   - ROCm
 * Kokkos
   - OpenMP
   - CUDA
   - SYCL
 * ...
:::
::::::::::::::

:::
::: {.column width="45%"}
### How to Introspect Those?

 * Analyze applications based on those models;
 * Understand application performances;
 * Understand interactions between applications / compilers / run-times / system / hardware;
 * Influence/optimize application at any point:
   - writing,
   - optimization,
   - execution.

:::
::::::::::::::

# Low / High level Programming Models

## Low / High level Programming Models

First thing we can do is sort the programming models in three categories:

 * Low level: CUDA driver, OpenCL, ROCm, L0...
 * High level: Kokkos, Raja, SYCL, CUDA, OpenMP (intermediate level?)
 * Library Level: BLAS, LAPACK, FFT, Neural Networks, etc...

Library are based on the underlying low or high-level programming models

## Objective

What can we do that would have an impact on as many layer of the stack as possible:

 * Application writing,
 * Application optimization,
 * Application modeling/simulation,
 * Application monitoring,
 * Node resource management,
 * Platform management.

Focus on low-level programming models

# Low-Level Programming Models

## Model Centric Debugging / Tracing

Low level programming models usually rely on APIs:

 * Traceable (library preloading);
 * Injectable;
 * Usually task based, with clear dependencies;
 * State is reconstructible, on the fly or post-mortem;
 * Enables verification,
 * and modeling.

## Simulation

Accurate tracing and accessible modeling of low level programming models and
tasks allows for accurate simulation:

 * Scalability studies,
 * Performance extrapolation,
 * Performance debugging.

## High-level Programming Models Introspection

Low level programming models are a window into high-level programming models:

 * Debug high level programming models,
 * Optimization of high level programming model,
 * Could be used to override high level programming model's behavior.

## Node Resource Management

Live tracing of low level programming models can be used as input to solve control
issues at the node level. Especially correlated with other kind of performance
metric tracing.

 * Check progress/liveliness
 * Check appropriate utilization
 * Use as input for balancing power between Threads / Processes / Tasks / etc...

## Platform Management

Post mortem analysis of traces (and inputs?) of HPC applications can be used to make platform-wide decisions about resources allocations, driving:

 * Global power repartition between different applications,
 * Higher priority for efficient applications?
 * Anticipation of application power/resource usage allowing for better platform tuning (amount of cooling, application interferences avoiding)

# THAPI: Tracing Heterogeneous APIs

## THAPI GOALS

  * Programing-Model centric tracing
    - For each events, all the arguments are saved. 
    - Semantic preserved
  * Flexible 
    - Fine granualarity, you can enable/disable individual events tracing.
    - Trace can be read pragmaticaly plugins (C, Python, Ruby)
    - We provide tools calibrated to our needs as starting-blocks. 

  * Low/Raisonable overhead
    - In order of \~0.2us / event for 
    - In order of \~2us second / event reading


## THAPI is a Collection of Tracers

### Use low level tracing: Linux Tracing Toolkit Next Generation (LTTng):

 * Low Overhead
 * Binary format
 * Well maintained and established (used in industry leading data-centers)

### Supported APIs

 * OpenCL
 * Level Zero 
 * Cuda (WIP)

Open source at: https://xgitlab.cels.anl.gov/heteroflow/tracer

## Example

* Wrapping the API entry points to be able to reconstruct the context.
* SYCL example:

\footnotesize
```C++
#include <sycl.hpp>

int main() {
  sycl::default_selector selector;
  sycl::queue myQueue(selector);
  myQueue.submit([&](sycl::handler &cgh) {
    sycl::stream sout(1024, 256, cgh);
    cgh.single_task<class hello_world>([=]() {
      sout << "Hello, World!" << sycl::endl;
    });
  });
  return 0;
}
```
\normalsize

## OpenCL Tracing of SYCL
* 69 OpenCL calls:

\tiny
```babeltrace_opencl
cl:clCreateProgramWithIL_start: context: 0x2522780, il: 0x461ec0, length: 41996,
                                errcode_ret: 0x7ffd9763f8cc
cl:clCreateProgramWithIL_stop: program: 0x24ed980, errcode_ret_val: CL_SUCCESS
cl:clBuildProgram_start: program: 0x24ed980, num_devices: 0, device_list: 0x0,
                         options: 0x7ffd9763fdf0, pfn_notify: 0x0, user_data: 0x0,
                         device_list_vals: [], options_val: ""
cl_build:infos: program: 0x24ed980, device: 0x24394c0, build_status: CL_BUILD_SUCCESS,
                build_options: "", build_log: ""
cl_build:infos_1_2: program: 0x24ed980, device: 0x24394c0,
                    binary_type: CL_PROGRAM_BINARY_TYPE_EXECUTABLE
cl_build:infos_2_0: program: 0x24ed980, device: 0x24394c0,
                    build_global_variable_total_size: 0
cl:clBuildProgram_stop: errcode_ret_val: CL_SUCCESS
cl:clCreateKernel_start: program: 0x24ed980, kernel_name: 0x24ee890,
                         errcode_ret: 0x7ffd9763fe64,
                         kernel_name_val: "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11hello_world"
cl_arguments:kernel_info: kernel: 0x24ed170,
                          function_name: "_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11hello_world",
                          num_args: 9, context: 0x2522780, program: 0x24ed980, attibutes: ""
cl:clCreateKernel_stop: kernel: 0x24ed170, errcode_ret_val: CL_SUCCESS
```
\normalsize

## Example Tool: iprof
\tiny
```
$iprof ./a.out
[...]
Trace location: /home/tapplencourt/lttng-traces/iprof-20210408-204629

== Level0 ==
API calls | 1 Hostnames | 1 Processes | 1 Threads

                                  Name |     Time | Time(%) | Calls |  Average |      Min |      Max | Failed |
                        zeModuleCreate | 119.15ms |  91.68% |     1 | 119.15ms | 119.15ms | 119.15ms |      0 |
     zeCommandQueueExecuteCommandLists |   6.94ms |   5.34% |     2 |   3.47ms |   2.29ms |   4.65ms |      0 |
         zeCommandListAppendMemoryCopy |   1.22ms |   0.94% |     1 |   1.22ms |   1.22ms |   1.22ms |      0 |
                   zeCommandListCreate | 741.99us |   0.57% |     2 | 371.00us | 355.94us | 386.05us |      0 |
                      zeMemAllocDevice | 610.28us |   0.47% |     1 | 610.28us | 610.28us | 610.28us |      0 |
          zeCommandListCreateImmediate | 344.82us |   0.27% |     1 | 344.82us | 344.82us | 344.82us |      0 |
                  zeCommandQueueCreate | 214.52us |   0.17% |     1 | 214.52us | 214.52us | 214.52us |      0 |
                zeEventHostSynchronize | 180.92us |   0.14% |     3 |  60.31us | 272.00ns | 180.20us |      0 |
                  zeCommandListDestroy | 100.67us |   0.08% |     3 |  33.56us |  21.50us |  57.36us |      0 |
                         zeFenceCreate |  82.36us |   0.06% |     2 |  41.18us |  40.46us |  41.90us |      0 |
                         zeEventCreate |  73.89us |   0.06% |     2 |  36.95us |   3.92us |  69.97us |      0 |
                                 Total | 129.97ms | 100.00% |   123 |                                       3 |

Device profiling | 1 Hostnames | 1 Processes | 1 Threads | 1 Device pointers

                         Name |    Time | Time(%) | Calls | Average |     Min |     Max |
zeCommandListAppendMemoryCopy | 23.04us |  50.53% |     1 | 23.04us | 23.04us | 23.04us |
                  hello_world | 22.56us |  49.47% |     1 | 22.56us | 22.56us | 22.56us |
                        Total | 45.60us | 100.00% |     2 |

Explicit memory trafic | 1 Hostnames | 1 Processes | 1 Threads

                         Name |    Byte | Byte(%) | Calls | Average |     Min |     Max |
             zeMemAllocDevice | 400.00B |  50.00% |     1 | 400.00B | 400.00B | 400.00B |
zeCommandListAppendMemoryCopy | 400.00B |  50.00% |     1 | 400.00B | 400.00B | 400.00B |
                        Total | 800.00B | 100.00% |     2 |
``` 
\normalsize

# HPC Centric

 * Can mix backend in same apps
 * Event are configurable to lower overhead

```
          Name |     Time | Time(%) |   Calls |  Average |      Min |      Max |
clSetKernelArg |    3.82s |  20.40% | 6607872 | 578.00ns | 335.00ns |  45.94us |
[...]
         Total |   18.75s | 100.00% | 8147809 |
```

 * Mutlithread / Multiprocess have first class support

# Conclusion and Future Work

## Conclusion

Working, robust and efficient tracers:

 * Used for simulation purposes,
 * Used for lightweight profiling,
 * Used for kernel extractions,
 * Used for debugging.

## Future work

Deployment strategies and use-cases on HPC infrastructures:

 * Lightweight monitoring for continuous usage,
 * Enabling full tracing of distributed applications,
 * Develop specialized trace analysis tools.
 * Timeline support
 * CUDA summary

More tracers:
 * ROCm / Hip

## Futur work

