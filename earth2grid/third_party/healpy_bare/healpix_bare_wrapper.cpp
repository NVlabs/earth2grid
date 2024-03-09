#include <torch/extension.h>
#include "healpix_bare.c"


int return_1(void) {
    return 1;
}

template<typename Func>
torch::Tensor vectorize(Func func, int nside, torch::Tensor input) {
  auto output = torch::empty_like(input);
  auto accessor = input.accessor<int64_t, 1>();
  auto out_accessor = output.accessor<int64_t, 1>();

  for (int64_t i = 0; i < input.size(0); ++i) {
    out_accessor[i] = func(nside, accessor[i]); // Specify nside appropriately
  }

  return output;
}

torch::Tensor ring2nest_wrapper(int nside, torch::Tensor input) {
  return vectorize(&ring2nest, nside, input);
}

torch::Tensor nest2ring_wrapper(int nside, torch::Tensor input) {
  return vectorize(&nest2ring, nside, input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("return_1", &return_1, "First implementation.");
  m.def("ring2nest", &ring2nest_wrapper, "Element-wise ring2nest conversion");
  m.def("nest2ring", &nest2ring_wrapper, "Element-wise nest2ring conversion");

}
