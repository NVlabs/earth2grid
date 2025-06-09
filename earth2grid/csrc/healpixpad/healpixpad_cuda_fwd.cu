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
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAUtils.h>
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

template<typename VAL_T, bool CHANNELS_LAST>
__global__ void HEALPixPadFwd_bulk_vec_k(
        const int padSize,
        const int dimI, const int dimJ,
        const int dimK, const int dimL, const int dimM,
        torch::PackedTensorAccessor32<VAL_T,5,torch::RestrictPtrTraits> vin,
        torch::PackedTensorAccessor32<VAL_T,5,torch::RestrictPtrTraits> vout)
{
  using VecT = typename VecTraits<VAL_T>::VecT;
  constexpr int W = VecTraits<VAL_T>::LANE_WIDTH;

  const long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= ((long long)dimI)*dimJ*dimK*dimL*dimM / W) {
    return;
  }

  const int dimKVec = CHANNELS_LAST ? dimK / W : dimK;
  const int dimMVec = CHANNELS_LAST ? dimM : dimM / W;

  int i,j,k,l,m;
  if constexpr(CHANNELS_LAST) {
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

  const VecT srcVec = *reinterpret_cast<const VecT*>(
              &getElem<VAL_T,CHANNELS_LAST>(vin,i,j,k,l,m));

  if (!CHANNELS_LAST && ((padSize & (W - 1)) != 0)) {
    // Store unvectorized as padding makes output unaligned
    const VAL_T* lanes = reinterpret_cast<const VAL_T*>(&srcVec);
#pragma unroll
    for (int w = 0; w < W; ++w) {
        getElemMutable<VAL_T,CHANNELS_LAST>(
            vout, i, j, k, padSize + l, padSize + m + w) = lanes[w];
    }
  } else {
    // src and dst are both aligned
    VecT* dstVec = reinterpret_cast<VecT*>(
      &getElemMutable<VAL_T,CHANNELS_LAST>(vout,i,j,k, padSize+l, padSize+m));
    *dstVec = srcVec;
  }
}

template<typename VAL_T, bool CHANNELS_LAST>
__global__ void HEALPixPadFwd_bulk_k(const int padSize,
				     const int dimI,
				     const int dimJ,
				     const int dimK,
				     const int dimL,
				     const int dimM,
				     const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vin,
				     torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  if (tid >= ((long long)dimI)*dimJ*dimK*dimL*dimM) {
    return;
  }

  // compute individual indices
  int i,j,k,l,m;
  if constexpr(CHANNELS_LAST) {
    k = tid % dimK;
    m = (tid / dimK) % dimM;
    l = (tid / (dimK * dimM)) % dimL;
    j = (tid / (dimK * dimM * dimL)) % dimJ;
    i = tid / (dimK * dimM * dimL * dimJ);
  } else {
    m = (tid % (dimM*dimL)) % dimM;
    l = (tid % (dimM*dimL)) / dimM;
    k = (tid % (dimM*dimL*dimK)) / (dimM*dimL);
    j = (tid % (dimM*dimL*dimK*dimJ)) / (dimM*dimL*dimK);
    i = (tid / (dimJ * dimK * dimL * dimM));
  }

  // copy data
  getElemMutable<VAL_T, CHANNELS_LAST>(vout, i, j, k, padSize+l, padSize+m) = getElem<VAL_T, CHANNELS_LAST>(vin, i, j, k, l, m);

  return;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getT_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
			const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
      // north faces
    case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, m, p); break;
    case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, m, p); break;
    case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, m, p); break;
    case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, m, p); break;
      // center faces
    case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, dimL-1-p, m); break;
    case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, dimL-1-p, m); break;
    case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, dimL-1-p, m); break;
    case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, dimL-1-p, m); break;
      // south faces
    case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 5, k, dimL-1-p, m); break;
    case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 6, k, dimL-1-p, m); break;
    case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 7, k, dimL-1-p, m); break;
    case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 4, k, dimL-1-p, m); break;
    }

  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getB_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
			const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
      // north faces
    case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  4, k, p, m); break;
    case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  5, k, p, m); break;
    case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  6, k, p, m); break;
    case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  7, k, p, m); break;
      // center faces
    case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, p, m); break;
    case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, p, m); break;
    case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, p, m); break;
    case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, p, m); break;
      // south faces
    case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, m, dimM-1-p); break;
    case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, m, dimM-1-p); break;
    case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, m, dimM-1-p); break;
    case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, m, dimM-1-p); break;
    }

  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getL_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
			const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

    switch(faceId) {
      // north faces
    case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, p, m); break;
    case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, p, m); break;
    case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, p, m); break;
    case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, p, m); break;
      // center faces
    case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, m, dimM-1-p); break;
    case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, m, dimM-1-p); break;
    case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, m, dimM-1-p); break;
    case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, m, dimM-1-p); break;
      // south faces
    case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 4, k, m, dimM-1-p); break;
    case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 5, k, m, dimM-1-p); break;
    case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 6, k, m, dimM-1-p); break;
    case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 7, k, m, dimM-1-p); break;
    }

  return ret;
}

  template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getR_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int sphrId,
      const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

    switch(faceId) {
      // north faces
    case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  5, k, m, p); break;
    case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  6, k, m, p); break;
    case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  7, k, m, p); break;
    case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  4, k, m, p); break;
      // center faces
    case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, m, p); break;
    case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, m, p); break;
    case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, m, p); break;
    case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, m, p); break;
      // south faces
    case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, dimL-1-p, m); break;
    case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, dimL-1-p, m); break;
    case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, dimL-1-p, m); break;
    case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, dimL-1-p, m); break;
    }

  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getTL_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
			 const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  const int pinv = padSize-1 - p;
  const int qinv = padSize-1 - q;

  switch(faceId) {
    // north faces
  case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, pinv, qinv); break;
  case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, pinv, qinv); break;
  case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, pinv, qinv); break;
  case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, pinv, qinv); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: {
    int srcTRface;
    int srcBLface;
    switch(faceId) {
    case  4: srcTRface = 3; srcBLface = 0; break;
    case  5: srcTRface = 0; srcBLface = 1; break;
    case  6: srcTRface = 1; srcBLface = 2; break;
    case  7: srcTRface = 2; srcBLface = 3; break;
    }
    if (p == q)  {
      ret = (getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcTRface, k, 0, dimM-1-qinv) \
              + getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcBLface, k, dimL-1-pinv, 0)) / VAL_T(2);
      break;
    } else if (p > q)  {
      ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcTRface, k, p-1-q, dimM-1-qinv);
      break;
    } else  /* p < q*/ {
      ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcBLface, k, dimL-1-pinv, q-1-p);
      break;
    }
  }
    // south faces
  case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 0, k, dimL-pinv, -qinv-1); break;
  case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 1, k, dimL-pinv, -qinv-1); break;
  case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 2, k, dimL-pinv, -qinv-1); break;
  case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 3, k, dimL-pinv, -qinv-1); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getTR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  const int pinv = padSize-1 - p;

  switch(faceId) {
    // north faces
  case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  1, k, dimL-1-pinv, q); break;
  case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  2, k, dimL-1-pinv, q); break;
  case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  3, k, dimL-1-pinv, q); break;
  case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  0, k, dimL-1-pinv, q); break;
    // center faces
  case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  5, k, dimL-1-pinv, q); break;
  case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  6, k, dimL-1-pinv, q); break;
  case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  7, k, dimL-1-pinv, q); break;
  case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  4, k, dimL-1-pinv, q); break;
    // south faces
  case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, dimL-1-pinv, q); break;
  case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, dimL-1-pinv, q); break;
  case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, dimL-1-pinv, q); break;
  case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, dimL-1-pinv, q); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getBL_d(int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  const int qinv = padSize-1 - q;

  switch(faceId) {
    // north faces
  case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  3, k, p, dimM-1-qinv); break;
  case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  0, k, p, dimM-1-qinv); break;
  case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  1, k, p, dimM-1-qinv); break;
  case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  2, k, p, dimM-1-qinv); break;
    // center faces
  case  4: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  7, k, p, dimM-1-qinv); break;
  case  5: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  4, k, p, dimM-1-qinv); break;
  case  6: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  5, k, p, dimM-1-qinv); break;
  case  7: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  6, k, p, dimM-1-qinv); break;
    // south faces
  case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, p, dimM-1-qinv); break;
  case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, p, dimM-1-qinv); break;
  case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, p, dimM-1-qinv); break;
  case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, p, dimM-1-qinv); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__device__ VAL_T getBR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int sphrId,
       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> sphr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
    // north faces
  case  0: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, p, q); break;
  case  1: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, p, q); break;
  case  2: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, p, q); break;
  case  3: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, p, q); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: {
    int srcTRface;
    int srcBLface;
    switch(faceId) {
    case  4: srcTRface = 11; srcBLface =  8; break;
    case  5: srcTRface =  8; srcBLface =  9; break;
    case  6: srcTRface =  9; srcBLface = 10; break;
    case  7: srcTRface = 10; srcBLface = 11; break;
    }
    if (p == q)  {
      ret = (getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcTRface, k, p, dimM-1) \
              + getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcBLface, k, dimL-1, q)) / VAL_T(2);
      break;
    } else if (p > q)  {
      ret =  getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcTRface, k, p, dimM-(p-q));
      break;
    } else  /* p < q*/ {
      ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, srcBLface, k, dimL-(q-p), q);
      break;
    }
  }
    // south faces
  case  8: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 10, k, dimL-p, -1-q); break;
  case  9: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId, 11, k, dimL-p, -1-q); break;
  case 10: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  8, k, dimL-p, -1-q); break;
  case 11: ret = getElem<VAL_T, CHANNELS_LAST>(sphr, sphrId,  9, k, dimL-p, -1-q); break;
  }
  return ret;
}

template<typename VAL_T, bool CHANNELS_LAST>
__global__ void HEALPixPadFwd_haloSD_k(const int padSize,
				       const int dimI,
				       const int dimJ,
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> input,
				       torch::PackedTensorAccessor32<VAL_T, 5, torch::RestrictPtrTraits> output) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  if (tid >= ((long long)dimI)*dimJ*dimK*dimM*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimK*dimM*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimK*dimM*padSize)) / (dimK*dimM*padSize);

  const int dimLO = dimL + 2*padSize;
  const int dimMO = dimM + 2*padSize;

  int k, p, m;
  if constexpr (CHANNELS_LAST) {
    k =  tid % dimK;
    p = (tid / dimK)            % padSize;
    m = (tid / (dimK * padSize)) % dimM;
  } else {
    m =  tid % dimM;
    p = (tid / dimM)            % padSize;
    k = (tid / (dimM * padSize)) % dimK;
  }
  // copy top    face
  getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, padSize-1-p, padSize+m) = getT_d<VAL_T, CHANNELS_LAST>(k, p, m, dimL, dimM, faceId, sphrId, input);
  // copy bottom face
  getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, padSize+dimL+p, padSize+m) = getB_d<VAL_T, CHANNELS_LAST>(k, p, m, dimL, dimM, faceId, sphrId, input);
  // copy left   face
  getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, padSize+m, padSize-1-p) = getL_d<VAL_T, CHANNELS_LAST>(k, p, m, dimL, dimM, faceId, sphrId, input);
  // copy right  face
  getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, padSize+m, padSize+dimM+p) = getR_d<VAL_T, CHANNELS_LAST>(k, p, m, dimL, dimM, faceId, sphrId, input);

  // padSize is always <= dimM(L)
  // so there are always enough
  // threads to fully cover the
  // corners
  if (m < padSize) {
    const int q = m;
    getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, p, q) = getTL_d<VAL_T, CHANNELS_LAST>(padSize, p, q, k, dimL, dimM, faceId, sphrId, input);
    getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, p, dimMO-padSize+q) = getTR_d<VAL_T, CHANNELS_LAST>(padSize, p, q, k, dimL, dimM, faceId, sphrId, input);
    getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, dimLO-padSize+p, q) = getBL_d<VAL_T, CHANNELS_LAST>(padSize, p, q, k, dimL, dimM, faceId, sphrId, input);
    getElemMutable<VAL_T, CHANNELS_LAST>(output, sphrId, faceId, k, dimLO-padSize+p, dimMO-padSize+q) = getBR_d<VAL_T, CHANNELS_LAST>(padSize, p, q, k, dimL, dimM, faceId, sphrId, input);
  }

  return;
}


template<typename REAL_T, bool CHANNELS_LAST>
void HEALPixPadFwd(int padSize,
		   torch::Tensor input,
		   torch::Tensor output,
		   cudaStream_t stream) {

  const int dimI = input.size(0);
  const int dimJ = input.size(1);
  const int dimK = (CHANNELS_LAST ? input.size(4) : input.size(2));
  const int dimL = (CHANNELS_LAST ? input.size(2) : input.size(3));
  const int dimM = (CHANNELS_LAST ? input.size(3) : input.size(4));

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

  const int W = VecTraits<REAL_T>::LANE_WIDTH;
  const bool canVec = (((CHANNELS_LAST ? dimK : dimM) & (W-1)) == 0);
  const int nth_b = THREADS;

  if (canVec) {
    const int nbl_b  = DIV_UP(dimI*dimJ*dimK*dimL*dimM/W, nth_b);

    HEALPixPadFwd_bulk_vec_k<REAL_T,CHANNELS_LAST><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM,
              input.packed_accessor32<REAL_T,5,torch::RestrictPtrTraits>(),
              output.packed_accessor32<REAL_T,5,torch::RestrictPtrTraits>());

    CHECK_ERROR("HEALPixPadFwd_bulk_vec_k");
  } else {
    const int nbl_b  = DIV_UP(dimI*dimJ*dimK*dimL*dimM, nth_b);

    HEALPixPadFwd_bulk_k<REAL_T,CHANNELS_LAST><<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM,
              input.packed_accessor32<REAL_T,5,torch::RestrictPtrTraits>(),
              output.packed_accessor32<REAL_T,5,torch::RestrictPtrTraits>());

    CHECK_ERROR("HEALPixPadFwd_bulk_k");
  }

  // copy haloes
  const int nth_f = THREADS;
  const int nbl_f = DIV_UP(dimI*dimJ*dimK*dimM*padSize, nth_f);

  // this also takes care of the corners
  REAL_T* dataIn_d = input.data_ptr<REAL_T>();
  REAL_T* dataOut_d = output.data_ptr<REAL_T>();
  HEALPixPadFwd_haloSD_k<REAL_T, CHANNELS_LAST><<<nbl_f, nth_f, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM,
						      input.packed_accessor32<REAL_T, 5, torch::RestrictPtrTraits>(),
						      output.packed_accessor32<REAL_T, 5, torch::RestrictPtrTraits>());

  CHECK_ERROR("HEALPixPadFwd_haloTB_k");

  return;
}


std::vector<torch::Tensor> healpixpad_cuda_forward(torch::Tensor input, int pad, bool channels_last) {

  const auto batch_size = input.size(0);
  const auto num_faces = input.size(1);
  const auto num_channels = (channels_last ? input.size(4) : input.size(2));
  const auto face_size = input.size(3);

  // allocate output tensor
  torch::TensorOptions options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
  torch::Tensor output;
  if(!channels_last) {
    output = torch::empty({batch_size, num_faces, num_channels, face_size+2*pad, face_size+2*pad}, options);
  } else {
    output = torch::empty({batch_size, num_faces, face_size+2*pad, face_size+2*pad, num_channels}, options);
  }

  // get cuda stream:
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // call wrapper
  switch (input.scalar_type()) {
  case torch::ScalarType::Double:
    if (channels_last) HEALPixPadFwd<double, true>(pad, input, output, stream);
    else HEALPixPadFwd<double, false>(pad, input, output, stream);
    break;
  case torch::ScalarType::Float:
    if (channels_last) HEALPixPadFwd<float, true>(pad, input, output, stream);
    else HEALPixPadFwd<float, false>(pad, input, output, stream);
    break;
  case torch::ScalarType::Half:
    if (channels_last) HEALPixPadFwd<at::Half, true>(pad, input, output, stream);
    else  HEALPixPadFwd<at::Half, false>(pad, input, output, stream);
    break;
  case torch::ScalarType::BFloat16:
    if (channels_last) HEALPixPadFwd<at::BFloat16, true>(pad, input, output, stream);
    else HEALPixPadFwd<at::BFloat16, false>(pad, input, output, stream);
    break;
  default:
    throw std::invalid_argument("Unsupported datatype for healpixpad_cuda_forward.");
  }

  return {output};
}
