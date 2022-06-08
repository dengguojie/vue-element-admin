#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
transdata constants
"""

from tbe.dsl.unify_schedule.constants import TransdataCategory as TC

DEFAULT = "default"
FP32_ALIGN_SIZE = 128
BLOCK = 32
FP16_BLOCK = 16
FP32_BLOCK = 8
INT8_BLOCK = 32

B32 = 4
B16 = 2
B8 = 1

FORWARD = 2
BACKWARD = 3
CONST_KEY = 123

STORAGE_ALIGN = 0
COMMON_ALIGN = 1
TRANSPOSE_NOT_WORK = 0
TRANSPOSE_WORK = 1
AVOID_CONFLICT_NOT_WORK = 0
AVOID_CONFLICT_WORK = 1

STRIDE_2 = 2
STRIDE_3 = 3
NO_OVERLAP = "no_overlap"

BASE_BRANCH = [TC.GENERAL_FORWARD, TC.GENERAL_BACKWARD]
BORROW_BRANCH = [TC.BORROW_N_B8B16_BACKWARD, TC.BORROW_N_B8B16_FORWARD,
                 TC.BORROW_H_B8B16_BACKWARD, TC.BORROW_H_B8B16_FORWARD]

# COMMON_ALIGN
COMMON_ALIGN_NEED_NODES = 4
# GENERAL\BN\BH
UB_CATEGORY_GENERAL = 0
UB_CATEGORY_BN = 1
UB_CATEGORY_BH = 2

CATEGORY_MAP_UB = {
    TC.GENERAL_FORWARD: UB_CATEGORY_GENERAL,
    TC.GENERAL_BACKWARD: UB_CATEGORY_GENERAL,
    TC.BORROW_N_B8B16_BACKWARD: UB_CATEGORY_BN,
    TC.BORROW_N_B8B16_FORWARD: UB_CATEGORY_BN,
    TC.BORROW_H_B8B16_FORWARD: UB_CATEGORY_BH,
    TC.BORROW_H_B8B16_BACKWARD: UB_CATEGORY_BH
}

# db is always 0, is_forward would be assured in compile
FORMAT = {
    "ub_category": "int",
    "shape_type": "int",
    "block_split_idx": "int",
    "ub_split_first_idx": "int",
    "ub_split_second_idx": "int",
    "block_factor": "int",
    "ub_first_factor": "int",
    "ub_second_factor": "int",
    "transpose_work": "int",
    "avoid_bank_conflict": "int",
    "block_dim": "int"
}

# decide kinds of cases
RULE = [("db", range(2), 10 ** 9),
        ("is_forward", [2, 3], 10 ** 8),
        ("ub_category", range(9), 10 ** 7),
        ("shape_type", range(9), 10 ** 6),
        ("block_split_idx", range(9), 10 ** 5),
        ("ub_split_first_idx", range(9), 10 ** 4),
        ("ub_split_second_idx", range(9), 10 ** 3),
        ("transpose_work", range(2), 10 ** 2),
        ("avoid_bank_conflict", range(2), 10 ** 1)]

# PAD-MODE
DO_TRANSPOSE_PAD = 2
DO_PAD = 1
