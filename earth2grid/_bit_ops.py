# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def compact_bits(bits):
    # Remove interleaved 0 bits
    bits = bits & 0x5555555555555555  # Mask: 01010101...
    # example implementation for 1 byte
    # 0a0b0c0d
    # 00ab00cd # (x | x >> 1)  & 00110011 = 0x33
    # 0000abcd # (x | x >> 2)  & 00001111 = 0x0F
    # --------
    # abc0d
    bits = (bits | (bits >> 1)) & 0x3333333333333333  # noqa
    bits = (bits | (bits >> 2)) & 0x0F0F0F0F0F0F0F0F  # noqa
    bits = (bits | (bits >> 4)) & 0x00FF00FF00FF00FF  # noqa
    bits = (bits | (bits >> 8)) & 0x0000FFFF0000FFFF  # noqa
    bits = (bits | (bits >> 16)) & 0x00000000FFFFFFFF  # noqa
    return bits


def spread_bits(bits):
    """
    bits is a 32 bit number (stored in int64)
    algorithm starts by moving the first 16 bits to the left by 16
    and proceeding recursively
    """
    # example implementation for a byte
    # 0000abcd
    # 00ab00cd # (x | x <<2)  & 00110011 = 0x33
    # 0a0b0c0d # (x | x <<1)  & 01010100 = 0x55
    # --------
    # abc0d
    bits = (bits | (bits << 16)) & 0x0000FFFF0000FFFF  # noqa
    bits = (bits | (bits << 8)) & 0x00FF00FF00FF00FF  # noqa
    bits = (bits | (bits << 4)) & 0x0F0F0F0F0F0F0F0F  # noqa
    bits = (bits | (bits << 2)) & 0x3333333333333333  # noqa
    bits = (bits | (bits << 1)) & 0x5555555555555555  # noqa
    return bits
