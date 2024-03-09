#include <torch/extension.h>
#include "healpix_bare.c"


int return_1(void) {
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("return_1", &return_1, "First implementation.");
}
