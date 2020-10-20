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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
# 'pylint: disable=R0903
class Pattern:
    """
    Pattern
    """
    ELEMWISE = "ElemWise"
    REDUCE = "CommReduce"
    OPAQUE = "Opaque"
    CONV2D = "Convolution"
    CONV2D_BACKPROP_INPUT = "Conv2d_backprop_input"
    CONV2D_BACKPROP_FILTER = "Conv2d_backprop_filter"


class CompileInfo:
    MAX_DTYPE = "_max_dtype_bytes"
    COEXISTING_QUANTITY = "_coexisting_quantity"
    UB_SIZE = "_ub_size"
    CORE_NUM = "_core_num"
    IS_SUPPORT_BROADCAST = "_is_support_broadcast"
    IS_SUPPORT_ABSORBABLE_BROADCAST = "_is_support_absorbable_broadcast"
    FUSION = "_fusion"
    IS_CONST_SHAPES = "_is_const_shapes"
    CONST_SHAPES = "_const_shapes"
    CONST_BLOCK_DIMS = "_const_block_dims"
    USE_SPECIAL_PATTERN = "_use_special_pattern"
    VARS = "_vars"

FAKE_NODE_TAG = "elewise_empty_intrin"

# <dsl insn, pass insn> mapping
INSN_MAPPING = {
    "elewise_binary_add": "vector_add",
    "elewise_binary_sub": "vector_sub",
    "elewise_binary_div": "vector_div",
    "elewise_binary_mul": "vector_mul",
    "elewise_binary_min": "vector_min",
    "elewise_binary_max": "vector_max",
    "elewise_binary_and": "vector_and",
    "elewise_binary_or": "vector_or",
    "elewise_single_log": "vector_ln",
    "elewise_single_exp": "vector_exp",
    "elewise_single_rec": "vector_rec",
    "elewise_single_VS_add": "vector_adds",
    "elewise_single_VS_mul": "vector_muls",
    "elewise_single_VS_max": "vector_maxs",
    "elewise_single_VS_min": "vector_mins",
    "elewise_single_abs": "vector_abs",
    "elewise_single_relu": "vector_relu",
    "elewise_single_not": "vector_not",
    "elewise_single_sqrt": "vector_sqrt",
    "elewise_single_rsqrt": "vector_rsqrt",
    "elewise_single_cast": "vector_conv",
    "elewise_single_ceil": "vector_conv_ceil",
    "elewise_single_floor": "vector_conv_floor",
    "elewise_single_trunc": "vector_conv_trunc",
    "elewise_single_round": "vector_conv_rint",
    "elewise_single_round_d": "vector_conv_round",
    "elewise_empty_intrin":  "phony_insn",
    "tuple_reduce_sum": "vector_reduce_sum",
    "reduce_sum":  "vector_reduce_sum",
    "reduce_min":  "vector_reduce_min",
    "reduce_max":  "vector_reduce_max",
    "reduce_prod":  "vector_reduce_prod",
    "broadcast":  "vector_broadcast",
    "unified_broadcast":  "vector_broadcast",
    "elewise_binary_cmpsel_gt": "vector_select_gt",
    "elewise_binary_cmpsel_ge": "vector_select_ge",
    "elewise_binary_cmpsel_lt": "vector_select_lt",
    "elewise_binary_cmpsel_le": "vector_select_le",
    "elewise_binary_cmpsel_eq": "vector_select_eq",
    "elewise_binary_cmpsel_ne": "vector_select_ne",
    "elewise_binary_vcmpv_gt": "vector_gt",
    "elewise_binary_vcmpv_ge": "vector_ge",
    "elewise_binary_vcmpv_lt": "vector_lt",
    "elewise_binary_vcmpv_le": "vector_le",
    "elewise_binary_vcmpv_eq": "vector_eq",
    "elewise_binary_vcmpv_ne": "vector_ne",
    "elewise_multiple_sel": "vector_select_bool",
}

# support scalar insn
# example: tensor - scalar
SUPPORT_SCALAR_INSNS = [
    "elewise_binary_add",
    "elewise_binary_sub",
    "elewise_binary_mul",
    "elewise_binary_div",
]

# need a block save scalar
NEED_TEMP_SPACE_INSNS = [
    "elewise_single_VS_max",
    "elewise_single_VS_min",
    "unknown_broadcast",
]

VSEL_INSNS = "elewise_multiple_sel"

VCMPSEL_INSNS = [
    "elewise_binary_cmpsel_gt",
    "elewise_binary_cmpsel_ge",
    "elewise_binary_cmpsel_lt",
    "elewise_binary_cmpsel_le",
    "elewise_binary_cmpsel_eq",
    "elewise_binary_cmpsel_ne",
]

BROADCAST_INSNS = [
    "broadcast",
    "unified_broadcast",
    "unknown_broadcast",
]

DTYPE_BYTE_MAPPING = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float16": 2,
    "int16": 2,
    "uint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}
