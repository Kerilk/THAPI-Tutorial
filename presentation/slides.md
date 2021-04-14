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
   - L0
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
 * Enables verification;
 * and modeling.

## Simulation

Accurate tracing and accessible modeling of low level programming models and
tasks allows for accurate simulation:

 * Scalability studies,
 * Performance extrapolation,
 * Performance debugging.

## High-level Programming Models Introspection

Low level programming models are windows into high-level programming models:

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

  * Programming-Model centric tracing
    - For each events, all the arguments are saved,
    - Semantic preserved.
  * Flexible 
    - Fine granularity, you can enable/disable individual events tracing,
    - Trace can be read programmatically (C, Python, Ruby),
    - We provide tools calibrated to our needs as starting-blocks. 

  * Low/Reasonable overhead

## THAPI Consist in 2 bigs components

Open source at: https://github.com/argonne-lcf/THAPI

  * The tracing of events
    - For each runtime calls, dump their arguments
    - Done via `LTTng`
  * The parsing of the trace 
    - Generate a summary, a pretty print, information for data-simulation...
    - Done via `babeltrace`

  By default the trace are dumped into disks.

## THAPI is a Collection of Tracers...

### Use low level tracing: Linux Tracing Toolkit Next Generation (LTTng):

 * Low Overhead
    - In order of \~0.2µs per traced event
 * Binary format
 * Well maintained and established (used in industry leading data-centers)

### Supported APIs

 * OpenCL
 * Level Zero 
 * CUDA (WIP)

## ...and a Collection of Tools to Parse your Traces

### Babeltrace 2

> Babeltrace 2 is the reference parser implementation of the Common Trace Format (CTF), a very versatile trace format followed by various tracers and tools such as LTTng and barectf. The Babeltrace 2 CLI,library and its Python bindings can read and write CTF traces. 

### Tools
 * Summary, timeline (experimental), pretty printing
 * Tools are developed in C, time to process an event is in order of \~2µs second per event

## THAPI Examples

\tiny
```fortran
> cat main.f90
PROGRAM target_teams_distribute_parallel_do
  implicit none
  INTEGER :: N0 = 32768
  INTEGER :: i0
  REAL, ALLOCATABLE :: src(:)
  REAL, ALLOCATABLE :: dst(:)
  ALLOCATE(dst(N0), src(N0) )
  CALL RANDOM_NUMBER(src)
  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO map(to: src) map(from: dst)
  DO i0 = 1, N0
    dst(i0) = src(i0)
  END DO
END PROGRAM target_teams_distribute_parallel_do
> ifx -fiopenmp -fopenmp-targets=spir64 main.F90 -o target_teams_distribute_parallel_do
```
\normalsize

## THAPI Examples: `tracer_opencl.sh` & `babeltrace_opencl`

Wrapping the API entry points to be able to reconstruct the context.

\tiny
```babeltrace_opencl
> tracer_opencl.sh ./target_teams_distribute_parallel_do # Using OpenCL backend
Traces will be output to /home/tapplencourt/lttng-traces/thapi-opencl-session-20210409-145006
[...]
> babeltrace_opencl /home/tapplencourt/lttng-traces/thapi-opencl-session-20210409-145006
[...]
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
cl:clCreateKernel_start: program: 0x24ed980, kernel_name: 0x24ee890, errcode_ret: 0x7ffd9763fe64,
                         kernel_name_val: "__omp_offloading_801_1a0d014_MAIN___l9"
cl_arguments:kernel_info: kernel: 0x24ed170,
                          function_name: "__omp_offloading_801_1a0d014_MAIN___l9",
                          num_args: 9, context: 0x2522780, program: 0x24ed980, attibutes: ""
cl:clCreateKernel_stop: kernel: 0x24ed170, errcode_ret_val: CL_SUCCESS
[...]
```
\normalsize

## THAPI Examples: iprof

\tiny
```
$iprof ./target_teams_distribute_parallel_do.out # Using Level0 backend
Trace location: /home/tapplencourt/lttng-traces/iprof-20210408-204629
API calls | 1 Hostnames | 1 Processes | 1 Threads
                             Name |     Time | Time(%) | Calls |  Average |      Min |      Max | Fail |
                   zeModuleCreate | 211.63ms |  90.48% |     1 | 211.63ms | 211.63ms | 211.63ms |    0 |
zeCommandQueueExecuteCommandLists |   9.38ms |   4.01% |     7 |   1.34ms | 576.87us |   3.77ms |    0 |
                 zeMemAllocDevice |   5.21ms |   2.23% |     4 |   1.30ms |   1.04ms |   1.44ms |    0 |
    zeCommandListAppendMemoryCopy |   4.48ms |   1.92% |     6 | 747.19us | 449.65us |   1.52ms |    0 |
        zeCommandQueueSynchronize |   1.52ms |   0.65% |     7 | 217.60us | 149.03us | 349.54us |    0 |
               zeCommandListReset | 609.80us |   0.26% |     7 |  87.11us |   2.40us | 439.86us |    0 |
             zeCommandQueueCreate | 218.66us |   0.09% |     1 | 218.66us | 218.66us | 218.66us |    0 |
              zeCommandListCreate | 149.12us |   0.06% |     1 | 149.12us | 149.12us | 149.12us |    0 |
  zeCommandListAppendLaunchKernel | 136.50us |   0.06% |     1 | 136.50us | 136.50us | 136.50us |    0 |
[...]
         zeModuleGetGlobalPointer |   1.30us |   0.00% |     1 |   1.30us |   1.30us |   1.30us |    1 |
                  zeKernelDestroy |   1.29us |   0.00% |     1 |   1.29us |   1.29us |   1.29us |    0 |
            zeKernelGetProperties |   1.13us |   0.00% |     1 |   1.13us |   1.13us |   1.13us |    0 |
                            Total | 233.90ms | 100.00% |   113 |                                     1 |

Device profiling | 1 Hostnames | 1 Processes | 1 Threads | 1 Device pointers
                                  Name |     Time | Time(%) | Calls | Average |     Min |     Max |
         zeCommandListAppendMemoryCopy | 177.60us |  56.86% |     6 | 29.60us | 22.56us | 38.56us |
            zeCommandListAppendBarrier |  70.40us |  22.54% |     1 | 70.40us | 70.40us | 70.40us |
__omp_offloading_801_1a0d014_MAIN___l9 |  64.32us |  20.59% |     1 | 64.32us | 64.32us | 64.32us |
                                 Total | 312.32us | 100.00% |     8 |

Explicit memory trafic | 1 Hostnames | 1 Processes | 1 Threads
                         Name |     Byte | Byte(%) | Calls | Average |    Min |      Max |
             zeMemAllocDevice | 262.29kB |  50.00% |     4 | 65.57kB | 72.00B | 131.07kB |
zeCommandListAppendMemoryCopy | 262.29kB |  50.00% |     6 | 43.71kB |  8.00B | 131.07kB |
                        Total | 524.58kB | 100.00% |    10 |
``` 
\normalsize

## HPC Centric

 * Can mix backend in same apps

\scriptsize
```bash
babeltrace2 ~/iprof-20210409-150449 |& grep ust_ze | wc -l
244
babeltrace2 ~/iprof-20210409-150449 |& grep ust_opencl | wc -l
53
```

 * Developed with Multiprocess / Multithread / MultiGPU in mind

 ```
iprof mpirun -n 2 ./a.out
 ```

 * Traced Event are configurable to adjust overhead

\tiny
```
          Name |     Time | Time(%) |   Calls |  Average |      Min |      Max |
clSetKernelArg |    3.82s |  20.40% | 6607872 | 578.00ns | 335.00ns |  45.94us |
[...]
         Total |   18.75s | 100.00% | 8147809 |
```
\normalsize


# Conclusion and Future Work

## Conclusion

Working, robust and efficient tracers:

 * Used for simulation purposes,
 * Used for lightweight profiling,
 * Used for kernel extractions,
 * Used for debugging.

Parsing Tools can be customized to your wills

## Future work

Deployment strategies and use-cases on HPC infrastructures:

 * Lightweight monitoring for continuous usage,
 * Enabling full tracing of distributed applications,
 * Develop specialized trace analysis tools.
 * Timeline support
 * CUDA summary

More tracers:

 * ROCm / Hip

