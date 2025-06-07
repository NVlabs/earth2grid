/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Written by Mauro Bisson <maurob@nvidia.com> and Thorsten Kurth <tkurth@nvidia.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> healpixpad_cuda_forward(
						   torch::Tensor input,
						   const int pad, const bool channels_last);

std::vector<torch::Tensor> healpixpad_cuda_backward(
						   torch::Tensor grad_input,
						   const int pad, const bool channels_last);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> healpixpad_forward(
					      torch::Tensor input,
					      const int pad, const bool channels_last){
  CHECK_INPUT(input);
  assert (pad >= 0);
  assert (input.dims() == 5);
  assert (input.size(1) == 12);
  assert (input.size(3) % 2 == 0);
  assert (pad < int(input.size(3)));

  if (!channels_last) {
    assert (input.size(3) == input.size(4));
  } else {
    assert (input.size(2) == input.size(3));
  }

  if (pad == 0) {
    return {input};
  } else {
    return healpixpad_cuda_forward(input, pad, channels_last);
  }
}

std::vector<torch::Tensor> healpixpad_backward(
					       torch::Tensor ginput,
					       const int pad, const bool channels_last){
  CHECK_INPUT(ginput);
  assert (ginput.dims() == 5);
  assert (ginput.size(1) == 12);
  assert (ginput.size(3) % 2 == 0);
  assert (pad >= 0);
  assert (pad < int(ginput.size(3)));

  if (!channels_last) {
    assert (ginput.size(3) == ginput.size(4));
  } else {
    assert (ginput.size(2) == ginput.size(3));
  }

  if (pad ==0) {
    return {ginput};
  } else {
    return healpixpad_cuda_backward(ginput, pad, channels_last);
  }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &healpixpad_forward, "HEALPixPad forward (CUDA)");
  m.def("backward", &healpixpad_backward, "HEALPixPad backward (CUDA)");
}
