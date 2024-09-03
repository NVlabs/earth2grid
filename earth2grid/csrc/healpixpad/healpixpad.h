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

#ifndef __HEALPIX_H__
#define __HEALPIX_H__

#ifdef __cplusplus
extern "C" {
#endif

// forward kernels
void HEALPixPad_fwd_fp32(int dimI, // batch size
			 int dimJ, // 12
			 int dimK, // no. of channels
			 int dimL, // face no. of rows
			 int dimM, // face no. of cols
			 float *dataIn_d,
			 float *dataOut_d,
			 cudaStream_t stream=0);

void HEALPixPad_fwd_fp64(int dimI, // batch size
			 int dimJ, // 12
			 int dimK, // no. of channels
			 int dimL, // face no. of rows
			 int dimM, // face no. of cols
			 double *dataIn_d,
			 double *dataOut_d,
			 cudaStream_t stream=0);

// backward kernels
void HEALPixPad_bwd_fp32(int dimI, // batch size
			 int dimJ, // 12
			 int dimK, // no. of channels
			 int dimL, // face no. of rows of dataOut_d (dataIn_d has dimL+2 rows)
			 int dimM, // face no. of cols of dataOut_d (dataIn_d has dimM+2 cols)
			 float *dataIn_d,
			 float *dataOut_d,
			 cudaStream_t stream=0);

void HEALPixPad_bwd_fp64(int dimI, // batch size
			 int dimJ, // 12
			 int dimK, // no. of channels
			 int dimL, // face no. of rows of dataOut_d (dataIn_d has dimL+2 rows)
			 int dimM, // face no. of cols of dataOut_d (dataIn_d has dimM+2 cols)
			 double *dataIn_d,
			 double *dataOut_d,
			 cudaStream_t stream=0);

#ifdef __cplusplus
}
#endif

#endif
