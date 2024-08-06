/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Written by Mauro Bisson <maurob@nvidia.com> and THorsten Kurth <tkurth@nvidia.com>
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

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

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
template<typename VAL_T>
__global__ void HEALPixPadFwd_bulk_k(const int padSize,
				     const int dimI,
				     const int dimJ,
				     const int dimK,
				     const int dimL,
				     const int dimM,
				     const VAL_T *__restrict__ vin,
				     VAL_T *__restrict__ vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  if (tid >= ((long long)dimI)*dimJ*dimK*dimL*dimM) {
    return;
  }

  const long long sliceId = tid / (dimM*dimL);

  const int i = (tid % (dimM*dimL)) / dimM;
  const int j = (tid % (dimM*dimL)) % dimM;

  const int dimLO = dimL + 2*padSize;
  const int dimMO = dimM + 2*padSize;

  vout[sliceId*dimMO*dimLO + (padSize+i)*dimMO + padSize+j] = vin[sliceId*dimL*dimM + i*dimM + j];

  return;
}

// faces functions

template<typename VAL_T>
__device__ VAL_T getTopFaceElem_d(const int k,
				  const int m,
				  const int dimL,
				  const int dimM,
				  const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + m];
}

template<typename VAL_T>
__device__ VAL_T getBottomFaceElem_d(const int k,
				     const int m,
				     const int dimL,
				     const int dimM,
				     const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + (dimL-1)*dimM + m];
}

template<typename VAL_T>
__device__ VAL_T getLeftFaceElem_d(const int k,
				   const int l,
				   const int dimL,
				   const int dimM,
				   const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + l*dimM];
}

template<typename VAL_T>
__device__ VAL_T getRightFaceElem_d(const int k,
				    const int l,
				    const int dimL,
				    const int dimM,
				    const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + l*dimM + dimM-1];
}

template<typename VAL_T>
__device__ VAL_T getT_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int faceLen,
			const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
    // north faces
  case  0: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 1*faceLen + p); break;
  case  1: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 2*faceLen + p); break;
  case  2: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 3*faceLen + p); break;
  case  3: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 0*faceLen + p); break;
    // center faces
  case  4: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 0*faceLen - p*dimM); break;
  case  5: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 1*faceLen - p*dimM); break;
  case  6: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 2*faceLen - p*dimM); break;
  case  7: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 3*faceLen - p*dimM); break;
    // south faces
  case  8: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 5*faceLen - p*dimM); break;
  case  9: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 6*faceLen - p*dimM); break;
  case 10: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 7*faceLen - p*dimM); break;
  case 11: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 4*faceLen - p*dimM); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getB_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int faceLen,
			const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
    // north faces
  case  0: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 4*faceLen + p*dimM); break;
  case  1: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 5*faceLen + p*dimM); break;
  case  2: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 6*faceLen + p*dimM); break;
  case  3: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 7*faceLen + p*dimM); break;
    // center faces
  case  4: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 11*faceLen + p*dimM); break;
  case  5: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr +  8*faceLen + p*dimM); break;
  case  6: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr +  9*faceLen + p*dimM); break;
  case  7: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 10*faceLen + p*dimM); break;
    // south faces
  case  8: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 11*faceLen - p); break;
  case  9: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr +  8*faceLen - p); break;
  case 10: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr +  9*faceLen - p); break;
  case 11: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 10*faceLen - p); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getL_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int faceLen,
			const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
    // north faces
  case  0: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 3*faceLen + p*dimM); break;
  case  1: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 0*faceLen + p*dimM); break;
  case  2: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 1*faceLen + p*dimM); break;
  case  3: ret = getTopFaceElem_d(k, m, dimL, dimM, sphrPtr + 2*faceLen + p*dimM); break;
    // center faces
  case  4: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 3*faceLen - p); break;
  case  5: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 0*faceLen - p); break;
  case  6: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 1*faceLen - p); break;
  case  7: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 2*faceLen - p); break;
    // south faces
  case  8: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 4*faceLen - p); break;
  case  9: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 5*faceLen - p); break;
  case 10: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 6*faceLen - p); break;
  case 11: ret = getRightFaceElem_d(k, m, dimL, dimM, sphrPtr + 7*faceLen - p); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getR_d(const int k,
			const int p,
			const int m,
			const int dimL,
			const int dimM,
			const int faceId,
			const int faceLen,
			const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  switch(faceId) {
    // north faces
  case  0: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 5*faceLen + p); break;
  case  1: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 6*faceLen + p); break;
  case  2: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 7*faceLen + p); break;
  case  3: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 4*faceLen + p); break;
    // center faces
  case  4: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr +  8*faceLen + p); break;
  case  5: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr +  9*faceLen + p); break;
  case  6: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 10*faceLen + p); break;
  case  7: ret = getLeftFaceElem_d(k, m, dimL, dimM, sphrPtr + 11*faceLen + p); break;
    // south faces
  case  8: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr +  9*faceLen - p*dimM); break;
  case  9: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 10*faceLen - p*dimM); break;
  case 10: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr + 11*faceLen - p*dimM); break;
  case 11: ret = getBottomFaceElem_d(k, m, dimL, dimM, sphrPtr +  8*faceLen - p*dimM); break;
  }
  return ret;
}

// corners functions

template<typename VAL_T>
__device__ VAL_T getTopLeftCornerElem_d(const int k,
					const int dimL,
					const int dimM,
					const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM];
}

template<typename VAL_T>
__device__ VAL_T getTopRightCornerElem_d(const int k,
					 const int dimL,
					 const int dimM,
					 const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + dimM-1];
}

template<typename VAL_T>
__device__ VAL_T getBottomLeftCornerElem_d(const int k,
					   const int dimL,
					   const int dimM,
					   const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + (dimL-1)*dimM];
}

template<typename VAL_T>
__device__ VAL_T getBottomRightCornerElem_d(const int k,
					    const int dimL,
					    const int dimM,
					    const VAL_T *__restrict__ facePtr) {

  return facePtr[k*dimL*dimM + dimL*dimM-1];
}

template<typename VAL_T>
__device__ VAL_T getCombCorner_d(const int k,
				 const int dimL,
				 const int dimM,
				 const VAL_T *__restrict__ facePtr0,
				 const VAL_T *__restrict__ facePtr1) {

  return (getTopRightCornerElem_d(k, dimL, dimM, facePtr0) +
	  getBottomLeftCornerElem_d(k, dimL, dimM, facePtr1)) / VAL_T(2);
}

template<typename VAL_T>
__device__ VAL_T getTL_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int faceLen,
			 const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  const int pinv = padSize-1 - p;
  const int qinv = padSize-1 - q;

  // offset from neighbor's corner
  // for non equatorial faces
  const int padOff = pinv*dimM + qinv;

  // offsets from neighbors' corners
  // for equatorial faces (there are
  // two, one for the left neigh, from
  // top right corner, and one for
  // the top neighbor, from the bottom left
  // corner)
  const int topRightPadOff   = (p-1 - q)*dimM - qinv;
  const int bottomLeftPadOff =     -pinv*dimM + (q-1 - p);

  switch(faceId) {
    // north faces
  case  0: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 2*faceLen + padOff); break;
  case  1: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 3*faceLen + padOff); break;
  case  2: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 0*faceLen + padOff); break;
  case  3: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 1*faceLen + padOff); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: {
    int srcTRface;
    int srcBLFace;
    switch(faceId) {
    case  4: srcTRface = 3; srcBLFace = 0; break;
    case  5: srcTRface = 0; srcBLFace = 1; break;
    case  6: srcTRface = 1; srcBLFace = 2; break;
    case  7: srcTRface = 2; srcBLFace = 3; break;
    }
    if (p == q)  {
      ret = getCombCorner_d(k, dimL, dimM,
			    sphrPtr + srcTRface*faceLen - qinv,
			    sphrPtr + srcBLFace*faceLen - pinv*dimM);
      break;
    } else if (p > q)  { ret =   getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + srcTRface*faceLen + topRightPadOff  ); break;
    } else  /* p < q*/ { ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + srcBLFace*faceLen + bottomLeftPadOff); break;
    }
  }
    // south faces
  case  8: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 0*faceLen - padOff); break;
  case  9: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 1*faceLen - padOff); break;
  case 10: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 2*faceLen - padOff); break;
  case 11: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 3*faceLen - padOff); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getTR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int faceLen,
			 const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  const int pinv = padSize-1 - p;
  const int padOff = -pinv*dimM + q;

  switch(faceId) {
    // north faces
  case  0: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 1*faceLen + padOff); break;
  case  1: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 2*faceLen + padOff); break;
  case  2: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 3*faceLen + padOff); break;
  case  3: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 0*faceLen + padOff); break;
    // center faces
  case  4: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 5*faceLen + padOff); break;
  case  5: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 6*faceLen + padOff); break;
  case  6: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 7*faceLen + padOff); break;
  case  7: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 4*faceLen + padOff); break;
    // south faces
  case  8: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr +  9*faceLen + padOff); break;
  case  9: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 10*faceLen + padOff); break;
  case 10: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + 11*faceLen + padOff); break;
  case 11: ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr +  8*faceLen + padOff); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getBL_d(int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int faceLen,
			 const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  const int qinv = padSize-1 - q;
  const int padOff = p*dimM - qinv;

  switch(faceId) {
    // north faces
  case  0: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 3*faceLen + padOff); break;
  case  1: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 0*faceLen + padOff); break;
  case  2: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 1*faceLen + padOff); break;
  case  3: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 2*faceLen + padOff); break;
    // center faces
  case  4: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 7*faceLen + padOff); break;
  case  5: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 4*faceLen + padOff); break;
  case  6: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 5*faceLen + padOff); break;
  case  7: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 6*faceLen + padOff); break;
    // south faces
  case  8: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 11*faceLen + padOff); break;
  case  9: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr +  8*faceLen + padOff); break;
  case 10: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr +  9*faceLen + padOff); break;
  case 11: ret = getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + 10*faceLen + padOff); break;
  }
  return ret;
}

template<typename VAL_T>
__device__ VAL_T getBR_d(const int padSize,
			 const int p,
			 const int q,
			 const int k,
			 const int dimL,
			 const int dimM,
			 const int faceId,
			 const int faceLen,
			 const VAL_T *__restrict__ sphrPtr) {

  VAL_T ret = VAL_T(0);

  // offset from neighbor's corner
  // for non equatorial faces
  const int padOff = p*dimM + q;

  // offsets from neighbors' corners
  // for equatorial faces (there are
  // two, one for the bottom neigh, from
  // top right corner, and one for
  // the right neighbor, from the bottom left
  // corner)
  const int topRightPadOff   =          p*dimM - (p-1 - q);
  const int bottomLeftPadOff = -(q-1 - p)*dimM + q;

  switch(faceId) {
    // north faces
  case  0: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr +  8*faceLen + padOff); break;
  case  1: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr +  9*faceLen + padOff); break;
  case  2: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 10*faceLen + padOff); break;
  case  3: ret = getTopLeftCornerElem_d(k, dimL, dimM, sphrPtr + 11*faceLen + padOff); break;
    // center faces
  case  4:
  case  5:
  case  6:
  case  7: {
    int srcTRface;
    int srcBLFace;
    switch(faceId) {
    case  4: srcTRface = 11; srcBLFace =  8; break;
    case  5: srcTRface =  8; srcBLFace =  9; break;
    case  6: srcTRface =  9; srcBLFace = 10; break;
    case  7: srcTRface = 10; srcBLFace = 11; break;
    }
    if (p == q)  {
      ret = getCombCorner_d(k, dimL, dimM,
			    sphrPtr + srcTRface*faceLen + p*dimM,
			    sphrPtr + srcBLFace*faceLen + q);
      break;
    } else if (p > q)  { ret =   getTopRightCornerElem_d(k, dimL, dimM, sphrPtr + srcTRface*faceLen + topRightPadOff  ); break;
    } else  /* p < q*/ { ret = getBottomLeftCornerElem_d(k, dimL, dimM, sphrPtr + srcBLFace*faceLen + bottomLeftPadOff); break;
    }
  }
    // south faces
  case  8: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 10*faceLen - padOff); break;
  case  9: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr + 11*faceLen - padOff); break;
  case 10: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr +  8*faceLen - padOff); break;
  case 11: ret = getBottomRightCornerElem_d(k, dimL, dimM, sphrPtr +  9*faceLen - padOff); break;
  }
  return ret;
}

template<typename VAL_T>
__global__ void HEALPixPadFwd_haloSD_k(const int padSize,
				       const int dimI,
				       const int dimJ,
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const VAL_T *__restrict__ vin,
				       VAL_T *__restrict__ vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  if (tid >= ((long long)dimI)*dimJ*dimK*dimM*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimK*dimM*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimK*dimM*padSize)) / (dimK*dimM*padSize);

  const int dimLO = dimL + 2*padSize;
  const int dimMO = dimM + 2*padSize;

  const long long faceLenI = ((long long)dimK)*dimL *dimM;
  const long long faceLenO = ((long long)dimK)*dimLO*dimMO;

  const VAL_T *__restrict__ sphrPtrI = vin  +  sphrId*dimJ          *faceLenI;
  VAL_T *__restrict__ facePtrO = vout + (sphrId*dimJ + faceId)*faceLenO;

  const int k = (tid / (padSize*dimM)) % dimK;
  const int p = (tid /          dimM)  % padSize;
  const int m =  tid                   % dimM;

  // copy top    face
  // copy bottom face
  // copy left   face
  // copy right  face
  facePtrO[k*dimLO*dimMO + (padSize-   1)*dimMO - p*dimMO + padSize+m] = getT_d(k, p, m, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO + (padSize+dimL)*dimMO + p*dimMO + padSize+m] = getB_d(k, p, m, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO + (padSize+   m)*dimMO    + padSize-   1 - p] = getL_d(k, p, m, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO + (padSize+   m)*dimMO    + padSize+dimM + p] = getR_d(k, p, m, dimL, dimM, faceId, faceLenI, sphrPtrI);

  // padSize is always <= dimM(L)
  // so there are always enough
  // threads to fully cover the
  // corners
  if (m < padSize) {
    const int q = m;
    facePtrO[k*dimLO*dimMO                                         + p*dimMO + q] = getTL_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
    facePtrO[k*dimLO*dimMO +                         dimMO-padSize + p*dimMO + q] = getTR_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
    facePtrO[k*dimLO*dimMO + (dimLO-padSize)*dimMO                 + p*dimMO + q] = getBL_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
    facePtrO[k*dimLO*dimMO + (dimLO-padSize)*dimMO + dimMO-padSize + p*dimMO + q] = getBR_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
  }

  return;
}

#if 0
template<typename VAL_T>
__global__ void HEALPixPadFwd_haloCR_k(const int padSize,
				       const int dimI,
				       const int dimJ,
				       const int dimK,
				       const int dimL,
				       const int dimM,
				       const VAL_T *__restrict__ vin,
				       VAL_T *__restrict__ vout) {

  const long long tid = ((long long)blockIdx.x)*blockDim.x + threadIdx.x;

  if (tid >= dimI*dimJ*dimK*padSize*padSize) {
    return;
  }

  const long long sphrId = tid / (dimJ*dimK*padSize*padSize);
  const long long faceId = (tid - sphrId*(dimJ*dimK*padSize*padSize)) / (dimK*padSize*padSize);

  const int dimLO = dimL + 2*padSize;
  const int dimMO = dimM + 2*padSize;

  const long long faceLenI = ((long long)dimK)*dimL *dimM;
  const long long faceLenO = ((long long)dimK)*dimLO*dimMO;

  const VAL_T *__restrict__ sphrPtrI = vin  +  sphrId*dimJ          *faceLenI;
  VAL_T *__restrict__ facePtrO = vout + (sphrId*dimJ + faceId)*faceLenO;

  const int k = (tid / (padSize*padSize)) % dimK;
  const int p = (tid /          padSize)  % padSize;
  const int q =  tid                      % padSize;

  // copy top-left corner
  // copy top-right corner
  // copy bottom-left corner
  // copy bottom-right corner
  facePtrO[k*dimLO*dimMO                                         + p*dimMO + q] = getTL_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO +                         dimMO-padSize + p*dimMO + q] = getTR_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO + (dimLO-padSize)*dimMO                 + p*dimMO + q] = getBL_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);
  facePtrO[k*dimLO*dimMO + (dimLO-padSize)*dimMO + dimMO-padSize + p*dimMO + q] = getBR_d(padSize, p, q, k, dimL, dimM, faceId, faceLenI, sphrPtrI);

  return;
}
#endif

template<typename REAL_T>
void HEALPixPadFwd(int padSize,
		   int dimI,
		   int dimJ,
		   int dimK,
		   int dimL,
		   int dimM,
		   REAL_T *dataIn_d,
		   REAL_T *dataOut_d,
		   cudaStream_t stream) {

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

  const int nth_b = THREADS;
  const int nbl_b = DIV_UP(dimI*dimJ*dimK*dimL*dimM, nth_b);

  HEALPixPadFwd_bulk_k<<<nbl_b, nth_b, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, dataIn_d, dataOut_d);
  CHECK_ERROR("HEALPixPadFwd_bulk_k");

  // copy haloes
  const int nth_f = THREADS;
  const int nbl_f = DIV_UP(dimI*dimJ*dimK*dimM*padSize, nth_f);

  // this also takes care of the corners
  HEALPixPadFwd_haloSD_k<<<nbl_f, nth_f, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, dataIn_d, dataOut_d);
  CHECK_ERROR("HEALPixPadFwd_haloTB_k");
  #if 0
  const int nth_c = THREADS;
  const int nbl_c = DIV_UP(dimI*dimJ*dimK*padSize*padSize, nth_c);

  HEALPixPadFwd_haloCR_k<<<nbl_c, nth_c, 0, stream>>>(padSize, dimI, dimJ, dimK, dimL, dimM, dataIn_d, dataOut_d);
  #endif
  CHECK_ERROR("HEALPixPadFwd_haloCR_k");
  //CHECK_CUDA(cudaStreamSynchronize(stream));

  return;
}

void HEALPixPad_fp32(
		     int padSize,
		     int dimI,
		     int dimJ,
		     int dimK,
		     int dimL,
		     int dimM,
		     float *dataIn_d,
		     float *dataOut_d,
		     cudaStream_t stream) {

  HEALPixPadFwd<float>(padSize, dimI, dimJ, dimK, dimL, dimM, dataIn_d, dataOut_d, stream);
  
  return;
}

void HEALPixPad_fp64(
		     int padSize,
		     int dimI,
		     int dimJ,
		     int dimK,
		     int dimL,
		     int dimM,
		     double *dataIn_d,
		     double *dataOut_d,
		     cudaStream_t stream) {

  HEALPixPadFwd<double>(padSize, dimI, dimJ, dimK, dimL, dimM, dataIn_d, dataOut_d, stream);

  return;
}


std::vector<torch::Tensor> healpixpad_cuda_forward(
						   torch::Tensor input,
						   int pad) {
  const auto batch_size = input.size(0);
  const auto num_faces = input.size(1);
  const auto num_channels = input.size(2);
  const auto face_size = input.size(3);
  int64_t shape[5] = {batch_size, num_faces, num_channels, face_size+2*pad, face_size+2*pad};

  // allocate output tensor
  c10::TensorOptions options = c10::TensorOptions().device(input.device()).dtype(input.dtype());
  torch::IntArrayRef size = c10::makeArrayRef<int64_t>(shape, 5);
  auto output = torch::empty(size, options);

  // get cuda stream:
  cudaStream_t my_stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
  
  switch (input.scalar_type()) {
  case torch::ScalarType::Double:
    HEALPixPadFwd<double>(
			  pad,
			  batch_size,
			  num_faces,
			  num_channels,
			  face_size,
			  face_size,
			  input.data_ptr<double>(),
			  output.data_ptr<double>(),
			  my_stream);
    break;
  case torch::ScalarType::Float:
    HEALPixPadFwd<float>(
			 pad,
			 batch_size,
			 num_faces,
			 num_channels,
			 face_size,
			 face_size,
			 input.data_ptr<float>(),
			 output.data_ptr<float>(),
			 my_stream);
    break;
  case torch::ScalarType::Half:
    HEALPixPadFwd<at::Half>(
			    pad,
			    batch_size,
			    num_faces,
			    num_channels,
			    face_size,
			    face_size,
			    input.data_ptr<at::Half>(),
			    output.data_ptr<at::Half>(),
			    my_stream);
    break;
  }

  return {output};
}

