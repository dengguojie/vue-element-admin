#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""
constants: for transdata classify
"""
DTYPE_BYTE = {
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "uint1": 0.125,
}

# reinterpret mapping
REINTERPRET_MAP = {"float32": "float16", "int32": "int16",
                   "uint32": "uint16", "bfloat32": "bfloat16",
                   "int64": "int16", "uint64": "uint16"}

UNKNOWN_DIM = -1
PACKET_SENDING_RATE = 256

GENERAL_FORWARD = "general.forward"
GENERAL_BACKWARD = "general.backward"
BORROW_N_B8B16_BACKWARD = "borrow.n.b8b16.backward"
BORROW_N_B8B16_FORWARD = "borrow.n.b8b16.forward"

# describe operation of pad
DO_NOTHING = 0
DO_PAD = 1
DO_TRANSPOSE_PAD = 2

# byte of type
B8 = 1
B16 = 2
B32 = 4
B64 = 8
BLOCK = 32
