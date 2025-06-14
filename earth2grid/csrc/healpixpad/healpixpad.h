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

#ifndef __HEALPIX_H__
#define __HEALPIX_H__

// Support vector lengths up to 128-bit
template<int BYTES>
struct RawVec { using type = void; };

template<> struct RawVec<2>  { using type = uint16_t; };
template<> struct RawVec<4>  { using type = uint32_t; };
template<> struct RawVec<8>  { using type = uint2; };
template<> struct RawVec<16> { using type = uint4; };

template<typename T, int W>
using VecT_t = typename RawVec<sizeof(T) * W>::type;

// CUDA vector wrapper struct providing basic element-wise arithmetic operations
template<typename T,int W>
struct VecW {
    using VecT = VecT_t<T,W>;
    VecT data;

    VecW() = default;

    __device__ VecW(const T& val) {
        #pragma unroll
        for (int i = 0; i < W; i++) {
            lane(i) = val;
        }
    }

    __device__ inline T lane(int i) const {
        return reinterpret_cast<const T*>(&data)[i];
    }
    __device__ inline T& lane(int i) {
        return reinterpret_cast<T*>(&data)[i];
    }

    __device__ VecW operator+(const VecW& b) const {
        VecW out;
#pragma unroll
        for(int i=0;i<W;++i) {
            out.lane(i)=lane(i)+b.lane(i);
        }
        return out;
    }
    __device__ VecW& operator+=(const VecW& b) {
#pragma unroll
        for(int i=0;i<W;++i) {
            lane(i) += b.lane(i);
        }
        return *this;
    }
    __device__ VecW operator*(T s) const {
        VecW out;
#pragma unroll
        for(int i=0;i<W;++i) {
            out.lane(i) = lane(i)*s;
        }
        return out;
    }
    __device__ VecW operator/(T s) const {
        VecW out;
#pragma unroll
        for(int i=0;i<W;++i) {
            out.lane(i) = lane(i)/s;
        }
        return out;
    }
};

// Checks whether datatype and vector length pair is allowed
template<typename S, int W>
constexpr bool vec_supported()
{
    return !std::is_same<VecT_t<S,W>, void>::value;
}

// Checks pointer alignment by bytes
template<size_t BYTES>
inline bool aligned(const void* p)
{
    static_assert(BYTES != 0, "Alignment must be non-zero.");
    if constexpr ((BYTES & (BYTES - 1)) == 0) {  // Check if BYTES is power of 2
        return (reinterpret_cast<std::uintptr_t>(p) & (BYTES - 1)) == 0;
    } else {
        return reinterpret_cast<std::uintptr_t>(p) % BYTES == 0;
    }
}

template<typename T>
__host__ int get_best_vector_width(
    bool channels_last,
    int dimK,
    int dimM,
    const void* input_ptr,
    const void* output_ptr)
{
    // Check vector widths in descending order with compile-time constants
    if (vec_supported<T, 8>() &&
        ((channels_last ? dimK : dimM) & 7) == 0 &&
        aligned<8*sizeof(T)>(input_ptr) &&
        aligned<8*sizeof(T)>(output_ptr)) {
        return 8;
    }

    if (vec_supported<T, 4>() &&
        ((channels_last ? dimK : dimM) & 3) == 0 &&
        aligned<4*sizeof(T)>(input_ptr) &&
        aligned<4*sizeof(T)>(output_ptr)) {
        return 4;
    }

    if (vec_supported<T, 2>() &&
        ((channels_last ? dimK : dimM) & 1) == 0 &&
        aligned<2*sizeof(T)>(input_ptr) &&
        aligned<2*sizeof(T)>(output_ptr)) {
        return 2;
    }

    return 1;  // Fallback to scalar
}

// Get const pointer for reading to element in a 5D (B, F, C, H, W) tensor
template<typename REAL_T, bool CHANNELS_LAST, int W = 1>
__device__ const VecW<REAL_T, W>* getElem(const torch::PackedTensorAccessor32<REAL_T, 5, torch::RestrictPtrTraits> sphr,
               const int i, const int j, const int k, const int l, const int m) {
    const REAL_T* ptr;
    if constexpr(CHANNELS_LAST) {
        ptr = &sphr[i][j][l][m][k]; // k (channels) is innermost dim
    } else {
        ptr = &sphr[i][j][k][l][m]; // m (spatial) is innermost dim
    }
    return reinterpret_cast<const VecW<REAL_T, W>*>(ptr);
}

// Mutable version of getElem for writing to element in a 5D (B, F, C, H, W) tensor
template<typename REAL_T, bool CHANNELS_LAST, int W = 1>
__device__ VecW<REAL_T, W>* getElemMutable(torch::PackedTensorAccessor32<REAL_T, 5, torch::RestrictPtrTraits> sphr,
                const int i, const int j, const int k, const int l, const int m) {
    REAL_T* ptr;
    if constexpr(CHANNELS_LAST) {
        ptr = &sphr[i][j][l][m][k];  // k (channels) is innermost dim
    } else {
        ptr = &sphr[i][j][k][l][m];  // m (spatial) is innermost dim
    }
    return reinterpret_cast<VecW<REAL_T, W>*>(ptr);
}


#endif
