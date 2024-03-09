#include <torch/extension.h>
#include "healpix_bare.c"


int return_1(void) {
    return 1;
}

template<typename Func>
auto wrap(Func func) {
  return [func](int nside, torch::Tensor input) {
    auto output = torch::empty_like(input);
    auto accessor = input.accessor<int64_t, 1>();
    auto out_accessor = output.accessor<int64_t, 1>();

    for (int64_t i = 0; i < input.size(0); ++i) {
      out_accessor[i] = func(nside, accessor[i]); // Specify nside appropriately
    }

    return output;
  };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("return_1", &return_1, "First implementation.");
  m.def("ring2nest", wrap(ring2nest), "Element-wise ring2nest conversion");
  m.def("nest2ring", wrap(nest2ring), "Element-wise nest2ring conversion");
}
