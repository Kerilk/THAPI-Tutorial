# First look at THAPI

ifx -fiopenmp -fopenmp-targets=spir64 simple.F90 -o target_teams_distribute_parallel_do
iprof ./target_teams_distribute_parallel_do

# Usage 1: Check backend error code

dpcpp error_handling.cpp -o error_handling.exe
#Everything look fine

./error_handling.exe
# But is it really?
iprof ./error_handling.exe

# Usage 2: Show trace and target specific call

icpx -fiopenmp -fopenmp-targets=spir64 memory_error.cpp -o memory_error.exe
./memory_error.exe
iprof ./memory_error.exe
iprof -t -r 
iprof -t -r  | grep clEnqueueMemcpyINTEL_entry

# Usage 3: Reconstruct the Semantic

dpcpp multi_context_usm.cpp -o multi_context_usm.exe
./multi_context_usm.exe
iprof ./multi_context_usm.exe 
iprof -t ./multi_context_usm.exe > log
# Kernel -> Program -> Context | Buffer -> Context 


