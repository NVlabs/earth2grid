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

template<typename T>
struct VecTraits {
    using VecT = T;
    static constexpr int LANE_WIDTH = 1;
};

template<>
struct VecTraits<double> {
    using VecT = double2;
    static constexpr int LANE_WIDTH = 2;
};

template<>
struct VecTraits<float> {
    using VecT = float4;
    static constexpr int LANE_WIDTH = 4;
};

template<>
struct VecTraits<at::Half> {
    using VecT = uint4;
    static constexpr int LANE_WIDTH = 8;
};

template<>
struct VecTraits<at::BFloat16> {
    using VecT = uint4;
    static constexpr int LANE_WIDTH = 8;
};

template<typename REAL_T, bool CHANNELS_LAST>
__device__ const REAL_T& getElem(const torch::PackedTensorAccessor32<REAL_T, 5, torch::RestrictPtrTraits> sphr,
               const int i, const int j, const int k, const int l, const int m) {
    if constexpr(CHANNELS_LAST) {
        return sphr[i][j][l][m][k];
    } else {
        return sphr[i][j][k][l][m];
    }
}

template<typename REAL_T, bool CHANNELS_LAST>
__device__ REAL_T& getElemMutable(torch::PackedTensorAccessor32<REAL_T, 5, torch::RestrictPtrTraits> sphr,
                const int i, const int j, const int k, const int l, const int m) {
    if constexpr(CHANNELS_LAST) {
        return sphr[i][j][l][m][k];
    } else {
        return sphr[i][j][k][l][m];
    }
}

#endif
