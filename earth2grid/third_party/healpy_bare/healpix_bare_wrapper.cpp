#include <vector>
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


template <typename Func>
auto wrap_2hpd(Func func) {

  return [func](int nside, torch::Tensor input){
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto x = torch::empty_like(input, options);
    auto y = torch::empty_like(input, options);
    auto f = torch::empty_like(input, options.dtype(torch::kInt32));

    auto output_options = torch::TensorOptions().dtype(torch::kInt64);
    auto output = torch::empty({input.size(0), 3}, output_options);
    auto out_accessor = output.accessor<int64_t, 2>();
    auto accessor = input.accessor<int64_t, 1>();

    for (int64_t i = 0; i < input.size(0); ++i) {
      auto hpd = func(nside, accessor[i]);
      out_accessor[i][0] = hpd.f;
      out_accessor[i][1] = hpd.y;
      out_accessor[i][2] = hpd.x;
    }

    return output;
  };
}

torch::Tensor hpd2loc_wrapper(int nside, torch::Tensor input) {
    auto accessor = input.accessor<int64_t, 2>();

    auto output_options = torch::TensorOptions().dtype(torch::kDouble);
    auto output = torch::empty({input.size(0), 3}, output_options);
    auto out_accessor = output.accessor<double, 2>();

    int64_t x, y, f;

    for (int64_t i = 0; i < input.size(0); ++i) {
      f = accessor[i][0];
      y = accessor[i][1];
      x = accessor[i][2];

      t_hpd hpd {x, y, f};
      tloc loc = hpd2loc(nside, hpd);
      out_accessor[i][0] = loc.z;
      out_accessor[i][1] = loc.s;
      out_accessor[i][2] = loc.phi;
    }

    return output;
}

// these are the minimal routines
// nest2hpd
// ring2hpd
// hpd2loc
// loc2ang




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("return_1", &return_1, "First implementation.");
  m.def("ring2nest", wrap(ring2nest), "Element-wise ring2nest conversion");
  m.def("nest2ring", wrap(nest2ring), "Element-wise nest2ring conversion");
  m.def("nest2hpd", wrap_2hpd(nest2hpd), "hpd is f, y ,x");
  m.def("ring2hpd", wrap_2hpd(ring2hpd), "hpd is f, y ,x");
  m.def("hpd2loc", &hpd2loc_wrapper, "loc is in z, s, phi");
}
