#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
utils const
"""

# def the mad pattern
GEMM_MODE = 0
GEVM_MODE = 1
CONV_MODE = 2

C0_SIZE = 16
ELEMENTS_VECTOR_OP_FP16 = 128

DEFAULT_MUL_VALUE = 1
DEFAULT_ADD_VALUE = 0

BLOCK_REDUCE = 16
# def the gemm int8/uint8 reduce const
BLOCK_REDUCE_INT8 = 32
# def the gemm/gevm vector const
BLOCK_VECTOR = 1
