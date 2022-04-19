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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""


from enum import Enum
from enum import auto


class Pattern:
    """
    Built-in Patterns
    """
    ELEMWISE = "ElemWise"
    BROADCAST = "Broadcast"
    REDUCE = "CommReduce"
    NORM = "Norm"
    TRANSPOSE = "Transpose"
    CONCAT = "Concat"
    SPLIT = "Split"
    OPAQUE = "Opaque"
    SOFTMAX = "Softmax"
    ASCEND_QUANT = "quant"
    ASCEND_ANTI_QUANT = "anti_quant"
    CONV2D = "Convolution"
    CONV2D_BACKPROP_INPUT = "Conv2d_backprop_input"
    CONV2D_BACKPROP_FILTER = "Conv2d_backprop_filter"
    MAT_MUL = "Matmul"
    BATCH_MATMUL = "BatchMatmul"
    LAYER_NORM_BETA_GAMMA_BACKPROP = "Layer_norm_beta_gamma_backprop"
    LAYER_NORM_BETA_GAMMA_BACKPROP_V2 = "Layer_norm_beta_gamma_backprop_v2"
    CONV3D = "conv3d"
    CONV3D_BACKPROP_INPUT = "Conv3d_backprop_input"
    CONV3D_BACKPROP_FILTER = "Conv3d_backprop_filter"
    SOFTMAX_CROSS_ENTROPY_WITH_LOGITS = "SoftmaxCrossEntropyWithLogits"
    LayerNorm = "LayerNorm"
    BN_TRAINING_UPDATE_GRAD = "BNTrainingUpdateGrad"
    LAYER_NORM_X_BACKPROP = "Layer_norm_x_backprop"
    LAYER_NORM_X_BACKPROP_V2 = "Layer_norm_x_backprop_v2"
    GATHER = "Gather"
    SLICE = "Slice"
    TRANSDATA = "Transdata"
    TUPLE_REDUCE = "TupleReduce"
    EXTRACT_IMAGE_PATCHES = "ExtractImagePatches"


class ElewisePattern:
    """
    Elewise sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    E_0 = "E_0"


class BroadcastPattern:
    """
    Broadcast sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    B_0 = "B_0"


class ReducePattern:
    """
    Reduce sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    R_0 = "R_0"


class ReduceCategory:
    """
    Category of Reduce
    """
    ALL_REDUCE = 1
    NOT_LAST_REDUCE = 2
    LAST_REDUCE = 3


class ReduceSchType:
    """
    Category of Reduce
    """
    NORMAL = 0
    PAD = 1
    TRANSPOSE = 2


class NormPattern:
    """
    Norm sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    N_0 = "N_0"


class GatherPattern:
    """
    Gather sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    NORMAL_SCHEDULE = "NORMAL_SCHEDULE"
    ZERO_SCHEDULE = "ZERO_SCHEDULE"


class SlicePattern:
    """
    Slice sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    NORMAL_SCHEDULE = "NORMAL_SCHEDULE"


class TransposePattern:
    """
    Transpose sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    T_0 = "T_0"


class TransdataCategory:
    """
    Category of Transdata
    """
    GENERAL_FORWARD = "general.forward"
    GENERAL_BACKWARD = "general.backward"
    BORROW_N_B8B16_BACKWARD = "borrow.n.b8b16.backward"
    BORROW_N_B8B16_FORWARD = "borrow.n.b8b16.forward"


class ConcatPattern:
    """
    Concat sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    C_0 = "C_0"


class SplitPattern:
    """
    Split sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    S_0 = "S_0"


class TupleReducePattern:
    """
    Tuple Reduce sub pattern.
    Each sub-pattern has a corresponding schedule, for function or performance.
    """
    # generic
    TR_0 = "TR_0"


class CompileInfo:
    """
    Built-in Compilation Info Keys
    """
    PATTERN = "_pattern"
    BASE_INFO = "_base_info"
    FLAG_INFO = "_flag_info"
    MAX_DTYPE = "_max_dtype_bytes"
    COEXISTING_QUANTITY = "_coexisting_quantity"
    UB_SIZE = "_ub_size"
    CORE_NUM = "_core_num"
    FUSION = "_fusion"
    CONST_SHAPES = "_const_shapes"
    CONST_BLOCK_DIMS = "_const_block_dims"
    VARS = "_vars"
    NORMAL_VARS = "_normal_vars"
    ATTR_VARS = "_attr_vars"
    CUSTOM_VARS = "_custom_vars"
    ELEWISE_VARS = "_elewise_vars"
    BLOCK_DIMS = "_block_dims"
    ATOMIC_FLAGS = "_atomic_flags"
    BROADCAST_AXIS = "_broadcast_axis"
    OUTS_UINT1 = "_outs_uint1"
    SOC_VERSION = "_soc_version"
    CONTAINS_ELEWISE_SCH = "_contains_elewise_sch"


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
    "elewise_single_lrelu": "vector_lrelu",
    "elewise_empty_intrin": "phony_insn",
    "tuple_reduce_sum": "vector_reduce_sum",
    "reduce_sum": "vector_reduce_sum",
    "reduce_min": "vector_reduce_min",
    "reduce_max": "vector_reduce_max",
    "reduce_prod": "vector_reduce_prod",
    "broadcast": "vector_broadcast",
    "unified_broadcast": "vector_broadcast",
    "set_value": "vector_dup",
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
    "elewise_binary_addrelu": "vector_addrelu",
    "elewise_binary_subrelu": "vector_subrelu",
    "elewise_multiple_sel": "vector_select_bool",
    "elewise_multiple_mla": "vector_mla",
    "elewise_multiple_madd": "vector_madd",
    "elewise_multiple_maddrelu": "vector_maddrelu",
    "elewise_binary_scalar_axpy": "vector_axpy",
    "dma_copy": "dma_copy"
}

TERNARY_INSNS = [
    "elewise_multiple_mla",
    "elewise_multiple_madd",
    "elewise_multiple_maddrelu",
    "elewise_binary_scalar_axpy",
]

# dst can not reuse src by insn realize
DST_SRC_NO_REUSE_SET = {
    # complex insns
    "elewise_binary_vcmpv_gt",
    "elewise_binary_vcmpv_ge",
    "elewise_binary_vcmpv_lt",
    "elewise_binary_vcmpv_le",
    "elewise_binary_vcmpv_eq",
    "elewise_binary_vcmpv_ne",
    "elewise_multiple_sel",
    "elewise_binary_cmpsel_gt",
    "elewise_binary_cmpsel_ge",
    "elewise_binary_cmpsel_lt",
    "elewise_binary_cmpsel_le",
    "elewise_binary_cmpsel_eq",
    "elewise_binary_cmpsel_ne",

    # cast insns
    "elewise_single_cast",
    "elewise_single_ceil",
    "elewise_single_floor",
    "elewise_single_trunc",
    "elewise_single_round",
    "elewise_single_round_d",
}

# support scalar insn
# example: tensor support scalar
SUPPORT_SCALAR_INSNS = [
    "elewise_binary_add",
    "elewise_binary_sub",
    "elewise_binary_mul",
    "elewise_binary_div",
    "elewise_binary_and",
    "elewise_binary_or",
    "elewise_binary_min",
    "elewise_binary_max",
    "elewise_single_log",
    "elewise_single_exp",
    "elewise_single_rec",
    "elewise_single_abs",
    "elewise_single_relu",
    "elewise_single_not",
    "elewise_single_sqrt",
    "elewise_single_rsqrt",
    "elewise_single_cast",
    "elewise_single_ceil",
    "elewise_single_floor",
    "elewise_single_trunc",
    "elewise_single_round",
    "elewise_single_round_d",
]

# need a block save scalar
NEED_TEMP_SPACE_INSNS = [
    "elewise_single_VS_max",
    "elewise_single_VS_min",
]

# need a node as temp space
NEED_EXTENT_NODE_INSNS = [
    "unknown_broadcast",
]

# need a block save scalar while dtype is s32
NEED_SPACE_WITH_DIFF_TYPE = [
    "elewise_single_VS_add",
    "elewise_single_VS_mul",
]

VCMP_INSNS = [
    "elewise_binary_vcmpv_gt",
    "elewise_binary_vcmpv_ge",
    "elewise_binary_vcmpv_lt",
    "elewise_binary_vcmpv_le",
    "elewise_binary_vcmpv_eq",
    "elewise_binary_vcmpv_ne",
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
    "uint1": 0.125,
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
    "bfloat16": 2,
}

AtomicSupportMap910 = {"support_dtype": ["float32", ],
                       "support_insn": ["reduce_sum", ], }

AtomicSupportMap920A = {"support_dtype": ["float32", "float16", "int32", "int16", "int8", "bfloat16"],
                        "support_insn": ["reduce_sum", "reduce_max", "reduce_min"]}


class ComputeType(Enum):
    """
    ComputeType
    """
    ANY = auto()
    UNKNOWN = auto()
    PLACEHOLDER = auto()
    ELEWISE = auto()
    BROADCAST = auto()
    REDUCE = auto()
    TRANSPOSE = auto()
    TRANSDATA = auto()
    SET_VALUE = auto()
    CONCAT = auto()
    SPLIT = auto()
    CAST = auto()
    CONV2D = auto()
    CONV2D_BP_INPUT = auto()
    CONV2D_BP_FILTER = auto()
    CONV3D_BP_INPUT = auto()
    CONV3D = auto()
    MAT_MUL = auto()
    CONV3D_BP_FILTER = auto()
    GATHER = auto()
    SLICE = auto()


COMPUTE_TYPE_INSN_MAPPING = {
    ComputeType.ELEWISE: {
        "elewise_binary_add", "elewise_binary_sub", "elewise_binary_div",
        "elewise_binary_mul", "elewise_binary_min", "elewise_binary_max",
        "elewise_binary_and", "elewise_binary_or", "elewise_binary_vcmpv_le",
        "elewise_binary_vcmpv_lt", "elewise_binary_vcmpv_ge",
        "elewise_binary_vcmpv_gt", "elewise_binary_vcmpv_ne",
        "elewise_binary_vcmpv_eq", "emit_insn_elewise_binary_cmp",
        "elewise_binary_logic", "elewise_single_log", "elewise_single_exp",
        "elewise_single_rec", "elewise_single_VS_add", "elewise_single_VS_mul",
        "elewise_single_VS_max", "elewise_single_VS_min", "elewise_single_abs",
        "elewise_single_relu", "elewise_single_not", "elewise_single_sqrt",
        "elewise_single_rsqrt", "elewise_single_lrelu", "elewise_multiple_mla",
        "elewise_multiple_madd", "elewise_multiple_maddrelu",
        "elewise_multiple_sel", "elewise_binary_scalar_axpy",
        "elewise_binary_cmpsel_gt", "elewise_binary_cmpsel_ge",
        "elewise_binary_cmpsel_lt", "elewise_binary_cmpsel_le",
        "elewise_binary_cmpsel_eq", "elewise_binary_cmpsel_ne",
        "elewise_binary_vcmpv_gt", "elewise_binary_vcmpv_ge",
        "elewise_binary_vcmpv_lt", "elewise_binary_vcmpv_le",
        "elewise_binary_vcmpv_eq", "elewise_binary_vcmpv_ne",
        "elewise_binary_addrelu", "elewise_binary_subrelu",
    },
    ComputeType.CAST: {
        "elewise_single_cast", "elewise_single_ceil", "elewise_single_floor",
        "elewise_single_trunc", "elewise_single_round", "elewise_single_round_d",
    },
    ComputeType.BROADCAST: {
        "unified_broadcast", "broadcast", "unknown_broadcast"
    },
    ComputeType.REDUCE: {
        "reduce_min", "reduce_max", "reduce_sum",
        "reduce_prod", "tuple_reduce_sum",
    },
    ComputeType.TRANSPOSE: {
        "transpose"
    },
    ComputeType.TRANSDATA: {
        "transdata"
    },
    ComputeType.SET_VALUE: {
        "set_value"
    },
    ComputeType.CONCAT: {
        "concat"
    },
    ComputeType.SPLIT: {
        "split"
    },
    ComputeType.CONV2D: {
        "conv_vector_remove_pad",
        "convolution_C",
        "convolution_C_UB",
        "convolution_c_col",
        "convolution_c_col_bias",
        "convolution_res_conv2d",
        "convolution_res_fp32_conv2d"
    },
    ComputeType.CONV2D_BP_INPUT: {
        "conv2d_backprop_input",
        "conv2d_backprop_input_opti"
    },
    ComputeType.CONV2D_BP_FILTER: {
        "conv2d_backprop_filterdw_ddr"
    },
    ComputeType.CONV3D: {
        "conv3d_fuse_fmap_tensor",
        "conv3d_c_col"
    },
    ComputeType.MAT_MUL: {
        "matmul",
        "gemm"
    },
    ComputeType.CONV3D_BP_INPUT: {
        "conv3d_backprop_input_c_ub"
    },
    ComputeType.CONV3D_BP_FILTER: {
        "conv3d_backprop_filterdw_ddr"
    },
    ComputeType.GATHER: {
        "gather",
        "gather_nd",
    },
    ComputeType.SLICE: {
        "slice"
    }
}
