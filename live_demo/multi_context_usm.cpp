#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int argc, char **argv) {

  const auto global_range = 10;

  sycl::default_selector selector;
  sycl::queue Q_mem(selector);
  int *A = sycl::malloc_shared<int>(global_range, Q_mem);

  sycl::queue Q_kernel(selector);
  Q_kernel.parallel_for(
        sycl::range<1>{sycl::range<1>(global_range)},
        [=](sycl::item<1> id) {
          A[id] = id;
  }).wait();

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
