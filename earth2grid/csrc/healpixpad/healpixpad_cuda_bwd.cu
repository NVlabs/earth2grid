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
#include <c10/cuda/CUDAStream.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cudamacro.h"
#include "healpixpad.h"

#define THREADS  64

#define DIV_UP(a,b) (((a)+((b)-1))/(b))

// All coordinates are w.r.t. a face[dimK][dimL][dimM]:
//
//     ^ k-axis
//    /
//   *---------*
//  /.        /|
// *---------*-+---> m-axis
// | .       | |
// | .       | |
// | *.......|.*
// |.        |/
// *---------*
// |
// |
// \/ l-axis
//
// Along the k-axis, dimJ=12 faces form a "sphere" and
// we have in total dimI sphere in the buffers
template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__global__ void HEALPixPadBck_bulk_k(const int padSize,
				     const int dimI,
				     const int dimJ,
				     const int dimK,
				     const int dimL,
				     const int dimM,
				     const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
             torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {
  using VecT = VecW<VAL_T, W>;

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;
  if (tid >= ((long long)dimI)*dimJ*dimK*dimL*dimM / W) {
    return;
  }

  const int dimKVec = CHANNELS_LAST ? dimK / W : dimK;
  const int dimMVec = CHANNELS_LAST ? dimM : dimM / W;

  // compute individual indices
  int i,j,k,l,m;
  if constexpr (CHANNELS_LAST) {
    k = (tid % dimKVec) * W;
    m = (tid / dimKVec) % dimM;
    l = (tid / (dimKVec * dimM)) % dimL;
    j = (tid / (dimKVec * dimM * dimL)) % dimJ;
    i =  tid / (dimKVec * dimM * dimL * dimJ);
  } else {
    m = (tid % dimMVec) * W;
    l = (tid / dimMVec) % dimL;
    k = (tid / (dimMVec * dimL)) % dimK;
    j = (tid / (dimMVec * dimL * dimK)) % dimJ;
    i =  tid / (dimMVec * dimL * dimK * dimJ);
  }

  VecT* dst = getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, i, j, k, l, m);

  if (!CHANNELS_LAST && ((padSize & (W - 1)) != 0)) {
    // Load unvectorized as padding breaks input alignment
    VecT tmp;
#pragma unroll
    for (int w = 0; w < W; ++w) {
        tmp.lane(w) = getElem<VAL_T, CHANNELS_LAST>(vin, i, j, k, padSize+l, padSize+m+w)->lane(0);
    }
    *dst = tmp;
  } else {
    // src and dst are both aligned
    *dst = *getElem<VAL_T, CHANNELS_LAST, W>(vin, i, j, k, padSize+l, padSize+m);
  }
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getT_d(const int padSize,
			const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
			const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 1, k, padSize+m, p); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 2, k, padSize+m, p); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 3, k, padSize+m, p); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 0, k, padSize+m, p); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 0, k, dimL-1-p, padSize+m); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 1, k, dimL-1-p, padSize+m); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 2, k, dimL-1-p, padSize+m); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 3, k, dimL-1-p, padSize+m); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 5, k, dimL-1-p, padSize+m); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 6, k, dimL-1-p, padSize+m); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 7, k, dimL-1-p, padSize+m); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 4, k, dimL-1-p, padSize+m); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getB_d(const int padSize,
			const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
			const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, p, padSize+m); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, p, padSize+m); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, p, padSize+m); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, p, padSize+m); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, p, padSize+m); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, p, padSize+m); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, p, padSize+m); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, p, padSize+m); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, padSize+m, dimM-1-p); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, padSize+m, dimM-1-p); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, padSize+m, dimM-1-p); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, padSize+m, dimM-1-p); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getL_d(const int padSize,
			const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
      const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  3, k, p, padSize+m); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  0, k, p, padSize+m); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  1, k, p, padSize+m); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  2, k, p, padSize+m); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  3, k, padSize+m, dimM-1-p); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  0, k, padSize+m, dimM-1-p); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  1, k, padSize+m, dimM-1-p); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  2, k, padSize+m, dimM-1-p); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, padSize+m, dimM-1-p); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, padSize+m, dimM-1-p); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, padSize+m, dimM-1-p); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, padSize+m, dimM-1-p); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getR_d(const int padSize,
			const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
      const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, padSize+m, p); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, padSize+m, p); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, padSize+m, p); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, padSize+m, p); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, padSize+m, p); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, padSize+m, p); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, padSize+m, p); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, padSize+m, p); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, dimL-1-p, padSize+m); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, dimL-1-p, padSize+m); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, dimL-1-p, padSize+m); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, dimL-1-p, padSize+m); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__global__ void HEALPixPadBck_haloTB_k(const int padSize,
				       const int dimI,
				       const int dimJ, // = 12
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
				       torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  const int dimKVec = CHANNELS_LAST ? dimK / W : dimK; // only vectorized for channels_last
  if (tid >= dimI*dimJ*dimKVec*dimM*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimKVec*dimM*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimKVec*dimM*padSize)) / (dimKVec*dimM*padSize);

  const int dimLI = dimL + 2*padSize;
  const int dimMI = dimM + 2*padSize;

  int k, p, m;
  if constexpr (CHANNELS_LAST) {
    k =  tid % dimKVec * W;
    p = (tid / dimKVec)             % padSize;
    m = (tid / (dimKVec * padSize)) % dimM;
  } else {
    m =  tid % dimM;
    p = (tid / dimM)             % padSize;
    k = (tid / (dimM * padSize)) % dimK;
  }

  // copy top    face
  // copy bottom face
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k,   padSize-1-p, m) \
    += getT_d<VAL_T, CHANNELS_LAST, W>(padSize, k, p, m, dimLI, dimMI, faceId, sphrId, vin);
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, dimL-padSize+p, m) \
    += getB_d<VAL_T, CHANNELS_LAST, W>(padSize, k, p, m, dimLI, dimMI, faceId, sphrId, vin);

  return;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__global__ void HEALPixPadBck_haloLR_k(const int padSize,
				       const int dimI,
				       const int dimJ, // = 12
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
               torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  const int dimKVec = CHANNELS_LAST ? dimK / W : dimK; // only vectorized for channels_last
  if (tid >= ((long long)dimI)*dimJ*dimKVec*dimM*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimKVec*dimM*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimKVec*dimM*padSize)) / (dimKVec*dimM*padSize);

  const int dimLI = dimL + 2*padSize;
  const int dimMI = dimM + 2*padSize;

  int k, p, m;
  if constexpr (CHANNELS_LAST) {
    k =  tid % dimKVec * W;
    p = (tid / dimKVec)             % padSize;
    m = (tid / (dimKVec * padSize)) % dimM;
  } else {
    m =  tid % dimM;
    p = (tid / dimM)             % padSize;
    k = (tid / (dimM * padSize)) % dimK;
  }

  // copy left   face
  // copy right  face
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, m, padSize-1-p) \
    += getL_d<VAL_T, CHANNELS_LAST, W>(padSize, k, p, m, dimLI, dimMI, faceId, sphrId, vin);
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, m, dimM-padSize+p) \
    += getR_d<VAL_T, CHANNELS_LAST, W>(padSize, k, p, m, dimLI, dimMI, faceId, sphrId, vin);

  return;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getTL_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
			 const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  const int pinv = padSize-1 - p;
  const int qinv = padSize-1 - q;

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 2, k, pinv, qinv); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 3, k, pinv, qinv); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 0, k, pinv, qinv); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 1, k, pinv, qinv); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 0, k, dimL-pinv, -qinv-1); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 1, k, dimL-pinv, -qinv-1); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 2, k, dimL-pinv, -qinv-1); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 3, k, dimL-pinv, -qinv-1); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getTR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  const int pinv = padSize-1 - p;

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  1, k, dimL-1-pinv, q); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  2, k, dimL-1-pinv, q); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  3, k, dimL-1-pinv, q); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  0, k, dimL-1-pinv, q); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, dimL-1-pinv, q); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, dimL-1-pinv, q); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, dimL-1-pinv, q); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, dimL-1-pinv, q); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, dimL-1-pinv, q); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, dimL-1-pinv, q); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, dimL-1-pinv, q); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, dimL-1-pinv, q); break;
  }

  if (p+q < padSize-1) {
    switch(faceId) {
      // north faces
    case  0: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, q+1+p, q); break;
    case  1: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, q+1+p, q); break;
    case  2: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, q+1+p, q); break;
    case  3: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, q+1+p, q); break;
    }
  }

  if (p == 0) {
    switch(faceId) {
      // north faces
    case  0: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, q, q) / VAL_T(2); break;
    case  1: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, q, q) / VAL_T(2); break;
    case  2: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, q, q) / VAL_T(2); break;
    case  3: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, q, q) / VAL_T(2); break;
    }
  }

  const int qinv = padSize-1 - q;
  if (p+q > padSize-1) {
    switch(faceId) {
      // south faces
    case  8: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, dimL-pinv, -(pinv+1+qinv)-1); break;
    case  9: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, dimL-pinv, -(pinv+1+qinv)-1); break;
    case 10: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, dimL-pinv, -(pinv+1+qinv)-1); break;
    case 11: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, dimL-pinv, -(pinv+1+qinv)-1); break;
    }
  }

  if (q == padSize-1) {
    switch(faceId) {
      // south faces
    case  8: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, dimL-pinv, -pinv-1) / VAL_T(2); break;
    case  9: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, dimL-pinv, -pinv-1) / VAL_T(2); break;
    case 10: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, dimL-pinv, -pinv-1) / VAL_T(2); break;
    case 11: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, dimL-pinv, -pinv-1) / VAL_T(2); break;
    }
  }

  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getBL_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  const int qinv = padSize-1 - q;

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  3, k, p, dimM-1-qinv); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  0, k, p, dimM-1-qinv); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  1, k, p, dimM-1-qinv); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  2, k, p, dimM-1-qinv); break;
    // center faces
  case  4: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, p, dimM-1-qinv); break;
  case  5: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, p, dimM-1-qinv); break;
  case  6: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, p, dimM-1-qinv); break;
  case  7: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, p, dimM-1-qinv); break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, p, dimM-1-qinv); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, p, dimM-1-qinv); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, p, dimM-1-qinv); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, p, dimM-1-qinv); break;
  }

  if (p+q < padSize-1) {
    switch(faceId) {
      // north faces
    case  0: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, p, p+q+1); break;
    case  1: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, p, p+q+1); break;
    case  2: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, p, p+q+1); break;
    case  3: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, p, p+q+1); break;
    }
  }

  if (q == 0) {
    switch(faceId) {
      // north faces
    case  0: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, p, p) / VAL_T(2); break;
    case  1: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, p, p) / VAL_T(2); break;
    case  2: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, p, p) / VAL_T(2); break;
    case  3: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, p, p) / VAL_T(2); break;
    }
  }

  if (p+q > padSize-1) {
    switch(faceId) {
      // south faces
    case  8: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, dimL-2*padSize+p+q+1, -qinv-1); break;
    case  9: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, dimL-2*padSize+p+q+1, -qinv-1); break;
    case 10: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, dimL-2*padSize+p+q+1, -qinv-1); break;
    case 11: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, dimL-2*padSize+p+q+1, -qinv-1); break;
    }
  }

  if (p == padSize-1) {
    switch(faceId) {
      // south faces
    case  8: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  4, k, dimL-qinv, -qinv-1) / VAL_T(2); break;
    case  9: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  5, k, dimL-qinv, -qinv-1) / VAL_T(2); break;
    case 10: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  6, k, dimL-qinv, -qinv-1) / VAL_T(2); break;
    case 11: ret += *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  7, k, dimL-qinv, -qinv-1) / VAL_T(2); break;
    }
  }

  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<VAL_T, W> getBR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  auto ret = VecW<VAL_T, W>(VAL_T(0));

  switch(faceId) {
    // north faces
  case  0: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, p, q); break;
  case  1: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, p, q); break;
  case  2: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, p, q); break;
  case  3: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, p, q); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: break;
    // south faces
  case  8: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 10, k, dimL-p, -1-q); break;
  case  9: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId, 11, k, dimL-p, -1-q); break;
  case 10: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  8, k, dimL-p, -1-q); break;
  case 11: ret = *getElem<VAL_T, CHANNELS_LAST, W>(sphr, sphrId,  9, k, dimL-p, -1-q); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
__global__ void HEALPixPadBck_haloCR_k(const int padSize,
				       const int dimI,
				       const int dimJ, // = 12
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
               torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  const int dimKVec = CHANNELS_LAST ? dimK / W : dimK; // only vectorized for channels_last
  if (tid >= ((long long)dimI)*dimJ*dimKVec*padSize*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimKVec*padSize*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimKVec*padSize*padSize)) / (dimKVec*padSize*padSize);

  const int dimLI = dimL + 2*padSize;
  const int dimMI = dimM + 2*padSize;

  int k, p, q;
  if constexpr (CHANNELS_LAST) {
    k =  tid % dimKVec * W;
    p = (tid / dimKVec)             % padSize;
    q = (tid / (dimKVec * padSize)) % padSize;
  } else {
    q =  tid % padSize;
    p = (tid / padSize)             % padSize;
    k = (tid / (padSize * padSize)) % dimK;
  }

  // copy top-left     corner
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, p, q) += getTL_d<VAL_T, CHANNELS_LAST, W>(padSize, p, q, k, dimLI, dimMI, faceId, sphrId, vin);
  // copy top-right    corner
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, p, dimM-padSize+q) += getTR_d<VAL_T, CHANNELS_LAST, W>(padSize, p, q, k, dimLI, dimMI, faceId, sphrId, vin);
  // copy bottom-left  corner
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, dimL-padSize+p, q) += getBL_d<VAL_T, CHANNELS_LAST, W>(padSize, p, q, k, dimLI, dimMI, faceId, sphrId, vin);
  // copy bottom-right corner
  *getElemMutable<VAL_T, CHANNELS_LAST, W>(vout, sphrId, faceId, k, dimL-padSize+p, dimM-padSize+q) += getBR_d<VAL_T, CHANNELS_LAST, W>(padSize, p, q, k, dimLI, dimMI, faceId, sphrId, vin);

  return;
}

template<typename VAL_T, bool CHANNELS_LAST, int W = 1>
void launch_halo_kernels(dim3 nbl_f, dim3 nth_f,
                        dim3 nbl_c, dim3 nth_c,
                        cudaStream_t stream,
                        int padSize, int dimI, int dimJ,
                        int dimK, int dimL, int dimM,
                        const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
                        torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {
  if constexpr (vec_supported<VAL_T, W>()) {
    HEALPixPadBck_haloTB_k<VAL_T, CHANNELS_LAST, W><<<nbl_f, nth_f, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, vin, vout);
    CHECK_ERROR("HEALPixPadBck_haloTB_k");
    HEALPixPadBck_haloLR_k<VAL_T, CHANNELS_LAST, W><<<nbl_f, nth_f, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, vin, vout);
    CHECK_ERROR("HEALPixPadBck_haloLR_k");
    HEALPixPadBck_haloCR_k<VAL_T, CHANNELS_LAST, W><<<nbl_c, nth_c, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, vin, vout);
    CHECK_ERROR("HEALPixPadBck_haloCR_k");
  }
}

template<typename REAL_T, bool CHANNELS_LAST>
void HEALPixPadBck(int padSize, torch::Tensor ginput, torch::Tensor goutput, cudaStream_t stream) {

  const int dimI = goutput.size(0);
  const int dimJ = goutput.size(1);
  const int dimK = (CHANNELS_LAST ? goutput.size(4) : goutput.size(2));
  const int dimL = (CHANNELS_LAST ? goutput.size(2) : goutput.size(3));
  const int dimM = (CHANNELS_LAST ? goutput.size(3) : goutput.size(4));

  if (dimI*dimJ*dimK*dimL*dimM <= 0) {
    fprintf(stderr, "%s:%d: error, one or more dimension is less than or equal zero!\n", __func__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (dimJ != 12) {
    fprintf(stderr, "%s:%d: error, dimJ must be equal to 12!\n", __func__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (dimL != dimM) {
    fprintf(stderr, "%s:%d: error, dimL must be equal to dimM!\n", __func__, __LINE__);
    exit(EXIT_FAILURE);
  }

  if (padSize > dimL) {
    fprintf(stderr, "%s:%d: error, padSize and less than or equal dimL (or dimM)\n", __func__, __LINE__);
    exit(EXIT_FAILURE);
  }

  // copy bulk
  const int bestW = get_best_vector_width<REAL_T>(
    CHANNELS_LAST,
    dimK,
    dimM,
    ginput.data_ptr(),
    goutput.data_ptr()
  );

  auto in = ginput.packed_accessor32<REAL_T, 5, torch::RestrictPtrTraits>();
  auto out = goutput.packed_accessor32<REAL_T, 5, torch::RestrictPtrTraits>();

  const int nth_b = THREADS;
  const int nbl_b = DIV_UP((dimI*dimJ*dimK*dimL*dimM)/bestW, nth_b);

  // Use constexpr safeguards to prevent compilation errors from unsupported vector types (i.e. double + width 8)
  switch (bestW) {
    case 8:
      if constexpr (vec_supported<REAL_T, 8>()) {
        HEALPixPadBck_bulk_k<REAL_T, CHANNELS_LAST, 8><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      }
      break;
    case 4:
      if constexpr (vec_supported<REAL_T, 4>()) {
        HEALPixPadBck_bulk_k<REAL_T, CHANNELS_LAST, 4><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      }
      break;
    case 2:
      if constexpr (vec_supported<REAL_T, 2>()) {
        HEALPixPadBck_bulk_k<REAL_T, CHANNELS_LAST, 2><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      }
      break;
    default:
      HEALPixPadBck_bulk_k<REAL_T, CHANNELS_LAST><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
    break;
  }

  CHECK_ERROR("HEALPixPadBck_bulk_k");


  // copy haloes
  const int bestW_halo = CHANNELS_LAST ? bestW : 1;
  const int nth_f = THREADS;
  const int nbl_f = DIV_UP((dimI*dimJ*dimK*dimM*padSize/bestW_halo), nth_f);

  const int nth_c = THREADS;
  const int nbl_c = DIV_UP((dimI*dimJ*dimK*padSize*padSize/bestW_halo), nth_c);

  switch (bestW_halo) {
    case 8:
      launch_halo_kernels<REAL_T, CHANNELS_LAST, 8>(nbl_f, nth_f, nbl_c, nth_c, stream, padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      break;
    case 4:
      launch_halo_kernels<REAL_T, CHANNELS_LAST, 4>(nbl_f, nth_f, nbl_c, nth_c, stream, padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      break;
    case 2:
      launch_halo_kernels<REAL_T, CHANNELS_LAST, 2>(nbl_f, nth_f, nbl_c, nth_c, stream, padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      break;
    default:
      launch_halo_kernels<REAL_T, CHANNELS_LAST>(nbl_f, nth_f, nbl_c, nth_c, stream, padSize, dimI, dimJ, dimK, dimL, dimM, in, out);
      break;
  }

  return;
}

std::vector<torch::Tensor> healpixpad_cuda_backward(torch::Tensor ginput, int pad, bool channels_last) {

  const auto batch_size = ginput.size(0);
  const auto num_faces = ginput.size(1);
  const auto num_channels = (channels_last ? ginput.size(4) : ginput.size(2));
  // the face size is the size of the output gradient
  const auto face_size = ginput.size(3) - 2*pad;

  // allocate output tensor
  torch::TensorOptions options = torch::TensorOptions().device(ginput.device()).dtype(ginput.dtype());
  torch::Tensor goutput;
  if (!channels_last) {
    goutput = torch::empty({batch_size, num_faces, num_channels, face_size, face_size}, options);
  } else {
    goutput = torch::empty({batch_size, num_faces, face_size, face_size, num_channels}, options);
  }

  // get cuda stream:
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  switch (ginput.scalar_type()) {
  case torch::ScalarType::Double:
    if (channels_last) HEALPixPadBck<double, true>(pad, ginput, goutput, stream);
    else HEALPixPadBck<double, false>(pad, ginput, goutput, stream);
    break;
  case torch::ScalarType::Float:
    if (channels_last) HEALPixPadBck<float, true>(pad, ginput, goutput, stream);
    else HEALPixPadBck<float, false>(pad, ginput, goutput, stream);
    break;
  case torch::ScalarType::Half:
    if (channels_last) HEALPixPadBck<at::Half, true>(pad, ginput, goutput, stream);
    else HEALPixPadBck<at::Half, false>(pad, ginput, goutput, stream);
    break;
  case torch::ScalarType::BFloat16:
    if (channels_last) HEALPixPadBck<at::BFloat16, true>(pad, ginput, goutput, stream);
    else HEALPixPadBck<at::BFloat16, false>(pad, ginput, goutput, stream);
    break;
  default:
    throw std::invalid_argument("Unsupported datatype for healpixpad_cuda_backward.");
  }

  return {goutput};
}
