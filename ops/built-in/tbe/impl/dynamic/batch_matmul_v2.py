# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic batch_matmul_v2
"""
import math
import warnings
from enum import Enum

from impl.util import fusion_util
from impl.util import util_gemm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_common import cal_mini_l1_size_matmul
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from tbe.common.context import op_context


# General limitation of the size for input shape: 2**31 - 1
SHAPE_SIZE_LIMIT = 2147483647
DYNAMIC_FLAG = -1
BLOCK_CUBE = 16
DYNAMIC_FLAG_UNRANK = [-2]
BATCH_NZ_LENGTH = 5
BATCH_ND_LENGTH = 3
ND_LENGTH = 2
L1FUSION_INPUT_CTR = 2
MKN_MIN = 1
LOWER_LIMIT_STR = "LOWER_LIMIT"


class Format(str, Enum):
    """
    class of format
    """
    FRACTAL_NZ = 'FRACTAL_NZ'
    ND = 'ND'


def get_op_support_info(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                        trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    get the batch_matmul_v2 split, which only split batch, m and n, cannot cut k with bias

    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    a_shape = input_x1.get("ori_shape")
    b_shape = input_x2.get("ori_shape")
    dtype_b = input_x2.get("dtype")
    if format_a == 'FRACTAL_NZ':
        trans_a = not trans_a
    if format_b == 'FRACTAL_NZ':
        trans_b = not trans_b

    batch_len_a = len(a_shape) - ND_LENGTH
    batch_len_b = len(b_shape) - ND_LENGTH
    if list(a_shape) == DYNAMIC_FLAG_UNRANK:
        batch_len_a = 1
    if list(b_shape) == DYNAMIC_FLAG_UNRANK:
        batch_len_b = 1

    # cut m
    if not trans_a:
        m_split_list = [0, [batch_len_a], [-1], [-1]]
        mk_split_list = [0, [batch_len_a + 1]]
    else:
        m_split_list = [0, [batch_len_a + 1], [-1], [-1]]
        mk_split_list = [0, [batch_len_a]]
    # cut n
    if not trans_b:
        n_split_list = [[1, [batch_len_b + 1], [-1], [-1]]]
        nk_split_list = [1, [batch_len_b]]
    else:
        n_split_list = [[1, [batch_len_b], [-1], [-1]]]
        nk_split_list = [1, [batch_len_b + 1]]

    if bias:
        axis_reduce_list = None
        n_split_list.append([2, [0], [-1], [-1]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                            util_select_op_base.ReduceOutput([0, 1, False])]]

    axis_split_matrix_batch = []
    for i in range(batch_len_a):
        batch_split_list = [[0, [i], [0], [0]]]
        if batch_len_b != 0:
            batch_split_list.append([1, [i], [0], [0]])
        axis_split_matrix_batch.append(
            [util_select_op_base.SplitInput(*batch_split_list),
             util_select_op_base.SplitOutput([0, [i]])]
        )

    axis_split_matrix_a = [
        [util_select_op_base.SplitInput(m_split_list),
         util_select_op_base.SplitOutput([0, [batch_len_a + 1]])]
    ]
    axis_split_matrix_b = [
        [util_select_op_base.SplitInput(*n_split_list),
         util_select_op_base.SplitOutput([0, [batch_len_a]])]
    ]

    axis_split_matrix = axis_split_matrix_a + axis_split_matrix_b + axis_split_matrix_batch
    min_l1space = cal_mini_l1_size_matmul(dtype_b)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def base_op_select_format(input_x, input_y, src_dtype, trans_b) -> tuple:
    """
    provide dynamic format to FE(Base processing)
    This funciton contains all basic format combinations

    return : dynamic format combination, static format combination
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    dyn_case_scenario_list = []
    full_case_scenario_list = []

    # The order from left to right is input1, input2, input3(bias), input4(offset_w), output
    base_case_scenario = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float16", "ND"),
                           ("int8", "ND"), ("float16", "FRACTAL_NZ"))]

    base_quant_case_scenario = [
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int8", "ND"), ("int32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("float16", "ND"), ("int8", "ND"), ("float16", "FRACTAL_NZ"))
    ]
    # Vector Logic
    fp32_dtype_scenario = [
        (("float", "NHWC"), ("float", "NHWC"), ("float", "NHWC"), ("int8", "ND"), ("float", "NHWC")),
        (("float", "ND"), ("float", "ND"), ("float", "ND"), ("int8", "ND"), ("float", "ND"))
    ]

    int32_dtype_scenario = [
        (("int32", "NHWC"), ("int32", "NHWC"), ("int32", "NHWC"), ("int8", "ND"), ("int32", "NHWC")),
        (("int32", "ND"), ("int32", "ND"), ("int32", "ND"), ("int8", "ND"), ("int32", "ND"))
    ]

    if not check_fp32_case_scenario(shape_a, shape_b, trans_b, src_dtype):
        fp32_dtype_scenario = []
        int32_dtype_scenario = []

    fp32_out_scenatio = [(("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"),
                          ("float", "ND"), ("int8", "ND"), ("float", "FRACTAL_NZ"))]
    rnn_scenatio = [
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float", "ND"), ("int8", "ND"),
         ("float", "FRACTAL_NZ")),
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float16", "ND"), ("int8", "ND"),
         ("float16", "FRACTAL_NZ"))]

    cube_vector_scenario = [
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_NZ"), ("float32", "ND"),
         ("int8", "ND"), ("float16", "FRACTAL_NZ")),
        (("float32", "FRACTAL_NZ"), ("float32", "FRACTAL_NZ"), ("float32", "ND"),
         ("int8", "ND"), ("float32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_NZ"), ("int32", "ND"), ("int8", "ND"), ("int32", "FRACTAL_NZ")),
        (("int8", "FRACTAL_NZ"), ("int8", "FRACTAL_Z"), ("int32", "ND"), ("int8", "ND"), ("int32", "FRACTAL_NZ")),
        (("bfloat16", "FRACTAL_NZ"), ("bfloat16", "FRACTAL_NZ"), ("float32", "ND"),
         ("int8", "ND"), ("bfloat16", "FRACTAL_NZ")),
        (("float16", "FRACTAL_NZ"), ("float16", "FRACTAL_ZN_RNN"), ("float32", "ND"),
         ("int8", "ND"), ("float16", "FRACTAL_NZ"))]
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    support_s322f32 = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "s322f32")

    dyn_case_scenario_list = base_case_scenario
    if dynamic_flag and not check_batch_range(input_x, input_y):
        warnings.warn("input_x, input_y out of batch_range")
        dyn_case_scenario_list = []

    # Construct scenario list for static
    if support_l0c2out:
        full_case_scenario_list = cube_vector_scenario
    else:
        full_case_scenario_list = base_case_scenario + base_quant_case_scenario + fp32_dtype_scenario + \
            fp32_out_scenatio + rnn_scenatio
        if support_s322f32:
            full_case_scenario_list += int32_dtype_scenario
    return dyn_case_scenario_list, full_case_scenario_list


def check_fp32_case_scenario(shape_a, shape_b, trans_b, src_dtype):
    """
    check if support float32 or int32 type

    Paramaters

    shape_a: list or tuple ,information of shape_a
    shape_b: list or tuple ,information of shape_b
    trans_b: bool
    src_type: type of input_x

    Returns

    support format for float32 or int32
    """
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)

    if not dynamic_flag:
        shape_a_length = len(shape_a)
        shape_b_length = len(shape_b)
        if shape_a_length != shape_b_length:
            return False
        elif shape_a_length == 2:
            return False
        elif trans_b:
            if shape_b[shape_a_length - 2] == 1:
                return False
        elif bool(1-trans_b):
            if shape_b[shape_a_length - 1] == 1:
                return False

    if _is_fuzzily_build() and src_dtype != "float16":
        return False

    return True


def _is_fuzzily_build():
    """
    check fuzzily build flag
    """
    context = op_context.get_context()
    return (context and context.get_build_type() == "fuzzily_build")


def check_batch_range(input_x, input_y):
    """
    Check the batch shape and range legal

    Parameters
    ----------
    input_x: dict with shape and range
    input_y: dict with shape and range

    Returns
    -------
    legit or not
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    if list(shape_a) == DYNAMIC_FLAG_UNRANK or list(shape_b) == DYNAMIC_FLAG_UNRANK:
        return True

    range_x1 = input_x.get("range")
    range_x2 = input_y.get("range")

    range_x1 = [[v, v] for v in shape_a] if not range_x1 and all(v > 0 for v in shape_a) else range_x1
    range_x2 = [[v, v] for v in shape_b] if not range_x2 and all(v > 0 for v in shape_b) else range_x2
    if len(shape_a) < ND_LENGTH:
        warnings.warn("shape_a length is at least 2-dimensional")
        return False
    if len(shape_b) < ND_LENGTH:
        warnings.warn("shape_b length is at least 2-dimensional")
        return False

    batch_range_x1 = range_x1[:(len(shape_a) - ND_LENGTH)]
    batch_range_x2 = range_x2[:(len(shape_b) - ND_LENGTH)]

    if (not batch_range_x1) or (not batch_range_x2):
        return True

    if len(batch_range_x1) != len(batch_range_x2):
        warnings.warn("shape_a and shape_b batch_range is not equal")
        return False

    return True


def gen_op_select_format_params(scenario_combinations: list, support_offset_w: bool = False) -> list:
    """
    generate format
    """
    input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                            datatype=','.join(
                                                x[0][0] for x in scenario_combinations),
                                            format=','.join(x[0][1] for x in scenario_combinations))
    input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                            datatype=','.join(
                                                x[1][0] for x in scenario_combinations),
                                            format=','.join(x[1][1] for x in scenario_combinations))
    # Bias supports only ND format
    input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                            datatype=','.join(
                                                x[2][0] for x in scenario_combinations),
                                            format=','.join(x[2][1] for x in scenario_combinations))
    if support_offset_w:
        input3 = util_select_op_base.gen_param(classify="input3", name="offset_w",
                                                datatype=','.join(
                                                    x[3][0] for x in scenario_combinations),
                                                format=','.join(x[3][1] for x in scenario_combinations))
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype=','.join(
                                                    x[4][0] for x in scenario_combinations),
                                                format=','.join(x[4][1] for x in scenario_combinations))
        param_list = [input0, input1, input2, input3, output0]
    else:
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                            datatype=','.join(
                                                x[3][0] for x in scenario_combinations),
                                            format=','.join(x[3][1] for x in scenario_combinations))
        param_list = [input0, input1, input2, output0]
    return param_list


def op_select_format(input_x, input_y, bias=None, offset_w=None, output_z=None, trans_a=False,
                     trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")
    scenario_combinations, _ = base_op_select_format(input_x, input_y, src_dtype, trans_b)

    param_list = gen_op_select_format_params(scenario_combinations, support_offset_w=True)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("BatchMatMulV2", op_mode="dynamic", support_fusion=False)
def batch_matmul_v2_fuse_compute(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                                 trans_a=False, trans_b=False, offset_x=0,
                                 kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: tensor or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        A dict object, dict with input tensor and output tensor
    """
    if output_z is None:
        output_z = {}

    fusion_util.check_fusion_input([input_x1])
    fusion_util.check_fusion_input([input_x2])
    if bias:
        fusion_util.check_fusion_input([bias])

    # set fusion build config
    build_cfg = tbe_register.get_fusion_buildcfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    if format_a == Format.FRACTAL_NZ:
        trans_a = not trans_a
    if format_b == Format.FRACTAL_NZ:
        trans_b = not trans_b

    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "kernel_name": kernel_name
    }
    op_res = tbe.gemm(input_x1, input_x2, para_dict)

    tensor_list = [input_x1, input_x2]
    if bias:
        tensor_list.append(bias)
    return {"op_placeholder": tensor_list, "op_res": [op_res]}


def _check_args(args: tuple, expect_args: list, msg: str) -> None:
    """
    check args
    """
    if args not in expect_args:
        error_manager_vector.raise_err_input_format_invalid(
            "mat_mul", msg, expect_args, args)


def _check_dynamic_mode_of_batch_matmul(shape_x1: tuple, shape_x2: tuple) -> None:
    """
    check dynamic mode
    """
    if len(shape_x1) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x1", "ori_shape dim must more than 1"
        )

    if len(shape_x2) < BATCH_ND_LENGTH - 1:
        error_manager_vector.raise_err_input_shape_invalid(
            "batch_matmul", "x2", "ori_shape dim must more than 1"
        )

    if all(i != DYNAMIC_FLAG for i in shape_x1) and all(i != DYNAMIC_FLAG for i in shape_x2):
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "dynamic must at least one of batch, m, k, n"
        )


def _check_dynamic_mode_of_matmul(shape_x1: tuple, shape_x2: tuple) -> None:
    """
    check dynamic mode
    """
    if len(shape_x1) != ND_LENGTH:
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x1", "ori_shape dim must be 2"
        )

    if len(shape_x2) != ND_LENGTH:
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x2", "ori_shape dim must be 2"
        )

    if all(i != DYNAMIC_FLAG for i in shape_x1) and all(i != DYNAMIC_FLAG for i in shape_x2):
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "dynamic must at least one in m,k,n"
        )


def _reset_range_by_shape(input_x1: dict, input_x2: dict, range_x1: tuple, range_x2: tuple) -> tuple:
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")
    # if all dim known, range may be empty
    range_x1 = tuple([dim, dim] for dim in shape_x1) if not range_x1 and all(dim > 0 for dim in shape_x1) else range_x1
    range_x2 = tuple([dim, dim] for dim in shape_x2) if not range_x2 and all(dim > 0 for dim in shape_x2) else range_x2
    # if shape is not -1, reset range as known
    if len(shape_x1) == len(range_x1):
        range_x1 = tuple(dim_range if shape_x1[idx] == -1 else (shape_x1[idx], shape_x1[idx])
                         for idx, dim_range in enumerate(range_x1))
    if len(shape_x2) == len(range_x2):
        range_x2 = tuple(dim_range if shape_x2[idx] == -1 else (shape_x2[idx], shape_x2[idx])
                         for idx, dim_range in enumerate(range_x2))
    return range_x1, range_x2


def _get_matmul_unrank_shape_and_range(input_x1: dict, input_x2: dict) -> list:
    shape_x1 = input_x1.get("ori_shape")
    shape_x2 = input_x2.get("ori_shape")
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    format_x1 = input_x1.get("format")
    format_x2 = input_x2.get("format")

    range_nd = ((1, None), (1, None))
    range_nz = ((1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    if list(shape_x1) == DYNAMIC_FLAG_UNRANK:
        shape_x1 = (-1, -1)
        range_x1 = range_nd if format_x1 == "ND" else range_nz
    if list(shape_x2) == DYNAMIC_FLAG_UNRANK:
        shape_x2 = (-1, -1)
        range_x2 = range_nd if format_x2 == "ND" else range_nz
    range_x1 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x1)
    range_x2 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x2)
    range_x1, range_x2 = _reset_range_by_shape(input_x1, input_x2, range_x1, range_x2)
    return [list(shape_x1), range_x1, list(shape_x2), range_x2]


def _get_batch_matmul_unrank_shape_and_range(input_x1: dict, input_x2: dict) -> list:
    shape_x1 = input_x1.get("ori_shape")
    shape_x2 = input_x2.get("ori_shape")
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    format_x1 = input_x1.get("format")
    format_x2 = input_x2.get("format")

    range_nd = ((1, None), (1, None), (1, None))
    range_nz = ((1, None), (1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    if list(shape_x1) == DYNAMIC_FLAG_UNRANK:
        shape_x1 = (-1, -1, -1)
        range_x1 = range_nd if format_x1 == "ND" else range_nz
    if list(shape_x2) == DYNAMIC_FLAG_UNRANK:
        shape_x2 = (-1, -1, -1)
        range_x2 = range_nd if format_x2 == "ND" else range_nz
    range_x1 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x1)
    range_x2 = tuple(dim_range if dim_range[0] >= MKN_MIN else (MKN_MIN, dim_range[1]) for dim_range in range_x2)
    return [list(shape_x1), range_x1, list(shape_x2), range_x2]


def _get_dynamic_shape_and_range(input_x1: dict, input_x2: dict, bias: dict, op_type: str) -> tuple:
    """
    get the shape and range of matmul
    """
    bias_range = None

    if op_type in ("MatMul", "MatMulV2"):
        shape_x1, range_x1, shape_x2, range_x2 = _get_matmul_unrank_shape_and_range(
            input_x1, input_x2)
    elif op_type in ("BatchMatMul", "BatchMatMulV2"):
        shape_x1, range_x1, shape_x2, range_x2 = _get_batch_matmul_unrank_shape_and_range(
            input_x1, input_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    if bias:
        bias_range = bias.get("range")

    return [shape_x1, shape_x2], [range_x1, range_x2, bias_range]


def _get_range_intersection(range1: list, range2: list, param_name: str, is_graph_mode: bool = False) -> list:
    """
    get range intersection of two range
    """
    if range1[1] is None:
        return range2
    if range2[1] is None:
        return range1

    range_ins = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    if range_ins[0] > range_ins[1]:
        if not is_graph_mode:
            reason = (f"the range of {param_name} is invalid because it has no intersection, "
                      "and the actual values are {range1}, {range2}")
            error_manager_vector.raise_err_specific_reson("mat_mul", reason)
        else:
            return LOWER_LIMIT_STR
    return range_ins


def _get_batch_range(range_x1: tuple, range_x2: tuple) -> list:
    """
    get range of batch
    """
    batch_range = [1, 1]
    range_x = []
    if range_x1 and range_x2:
        if len(range_x1) != len(range_x2):
            error_manager_vector.raise_err_specific_reson(
                "batch_matmul", "the batch length is not same of x1 and x2"
            )
        for range_mem1, range_mem2 in zip(range_x1, range_x2):
            range_ins = _get_range_intersection(range_mem1, range_mem2, "batch_range")
            range_x.append(range_ins)
    elif range_x2:
        range_x = range_x2
    else:
        range_x = range_x1

    for range_mem in range_x:
        if range_mem[1] is None:
            batch_range = [1, None]
            break
        else:
            batch_range[0] = min(batch_range[0] * range_mem[0], SHAPE_SIZE_LIMIT)
            batch_range[1] = min(batch_range[1] * range_mem[1], SHAPE_SIZE_LIMIT)

    return batch_range


def _get_input_x1_range(range_x1: tuple, format_x1: str, trans_a: bool, op_type: str, is_graph_mode: bool) -> list:
    range_len = BATCH_ND_LENGTH if format_x1 == 'ND' else BATCH_NZ_LENGTH
    if len(range_x1) >= range_len - 1:
        if format_x1 == 'FRACTAL_NZ':
            # trans_a True:  m1, k1, k0, m0
            # trans_a False: k1, m1, m0, k0
            k_x1_index = -4
            m_index = -3
            batch_range_x1 = range_x1[:-4]
        elif format_x1 == 'ND':
            m_index = -2
            k_x1_index = -1
            batch_range_x1 = range_x1[:-2]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    m_range = list(range_x1[m_index])
    k_range_x1 = list(range_x1[k_x1_index])
    if not is_graph_mode:
        operation.get_op_context().add_addition("batch_range_x1", batch_range_x1)
    if trans_a:
        m_range, k_range_x1 = k_range_x1, m_range
    return [m_range, k_range_x1, batch_range_x1]


def _get_input_x2_range(range_x2: tuple, format_x2: str, trans_b: bool, op_type: str, is_graph_mode: bool) -> list:
    range_len = BATCH_ND_LENGTH if format_x2 == 'ND' else BATCH_NZ_LENGTH
    if len(range_x2) >= range_len - 1:
        if format_x2 == 'FRACTAL_NZ':
            # trans_b True:  k1, n1, n0, k0
            # trans_b False: n1, k1, k0, n0
            n_index = -4
            k_x2_index = -3
            batch_range_x2 = range_x2[:-4]
        elif format_x2 == 'ND':
            k_x2_index = -2
            n_index = -1
            batch_range_x2 = range_x2[:-2]
        elif format_x2 == 'FRACTAL_Z':
            n_index = -3
            k_x2_index = -4
            batch_range_x2 = range_x2[:-4]
    else:
        error_manager_vector.raise_err_specific_reson(op_type, "Lenth of x1_range illegal")
    k_range_x2 = list(range_x2[k_x2_index])
    n_range = list(range_x2[n_index])
    if not is_graph_mode:
        operation.get_op_context().add_addition("batch_range_x2", batch_range_x2)
    if trans_b:
        k_range_x2, n_range = n_range, k_range_x2
    return [k_range_x2, n_range, batch_range_x2]


def _get_input_range(range_x1: tuple, format_x1: str, range_x2: tuple, format_x2: str, range_bias: tuple,
                     trans_a: bool, trans_b: bool, op_type: str, is_graph_mode: bool = False) -> list:
    """
    get range in batch, m, k, n and check range
    """
    batch_range_x1, batch_range_x2 = None, None
    if range_x1:
        m_range, k_range_x1, batch_range_x1 = _get_input_x1_range(range_x1, format_x1, trans_a, op_type, is_graph_mode)
    else:
        m_range = [1, None]
        k_range_x1 = [1, None]

    if range_x2:
        k_range_x2, n_range, batch_range_x2 = _get_input_x2_range(range_x2, format_x2, trans_b, op_type, is_graph_mode)
    else:
        k_range_x2 = [1, None]
        n_range = [1, None]

    k_range = _get_range_intersection(k_range_x1, k_range_x2, "k_range", is_graph_mode)
    if range_bias:
        range_bias_n = list(range_bias[0])
        if range_bias[0][1] is not None:
            if format_x2 in ('FRACTAL_NZ', 'FRACTAL_Z'):
                range_bias_n = [math.ceil(i / BLOCK_CUBE) for i in range_bias[0]]
        n_range = _get_range_intersection(n_range, range_bias_n, "n_range", is_graph_mode)

    # in fuzzy compile, if n/k's range has no intersection return LOWER_LIMIT
    wrong_range_flag = LOWER_LIMIT_STR in (n_range, k_range)
    if wrong_range_flag:
        return LOWER_LIMIT_STR

    # in generalization func of fuzzy compile, only need check. Not add_addition
    batch_range = None
    if not is_graph_mode:
        batch_range = _get_batch_range(batch_range_x1, batch_range_x2)

    return [batch_range, m_range, k_range, n_range]


def check_and_config_para(input_x1: dict, input_x2: dict, bias: dict, output_z: dict,
                          trans_a: bool, trans_b: bool, kernel_name: str, op_type: str) -> tuple:
    """
    check and config dynamic mode
    """
    # get format and dtype
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    format_out = output_z.get("format")
    dtype_a = input_x1.get("dtype").lower()
    dtype_b = input_x2.get("dtype").lower()
    dtype_out = output_z.get("dtype").lower()

    # check kernel_name dtype and format
    para_check.check_kernel_name(kernel_name)
    expect_args = [('FRACTAL_NZ', 'float16', 'FRACTAL_NZ', 'float16', 'FRACTAL_NZ', 'float16'),
                   ('ND', 'float16', 'ND', 'float16', 'ND', 'float16'),
                   ('ND', 'float16', 'FRACTAL_NZ', 'float16', 'ND', 'float16'),
                   ('FRACTAL_NZ', 'float16', 'FRACTAL_Z', 'float16', 'FRACTAL_NZ', 'float16'),
                   ('ND', 'float16', 'ND', 'float16', 'FRACTAL_NZ', 'float32'),
                   ('FRACTAL_NZ', 'float16', 'FRACTAL_NZ', 'float16', 'FRACTAL_NZ', 'float32')]
    _check_args((format_a, dtype_a, format_b, dtype_b, format_out, dtype_out),
                expect_args, "format_a, dtype_a, format_b, dtype_b, format_out, dtype_out")

    # get range and ori_shape
    shape_input, range_input = _get_dynamic_shape_and_range(input_x1, input_x2, bias, op_type)
    range_x1, range_x2, range_bias = range_input
    shape_x1, shape_x2 = shape_input

    # check dynamic mode
    if op_type in ("MatMul", "MatMulV2"):
        _check_dynamic_mode_of_matmul(shape_x1, shape_x2)
    elif op_type in ("BatchMatMul", "BatchMatMulV2"):
        _check_dynamic_mode_of_batch_matmul(shape_x1, shape_x2)
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)
    # get range in m,k,n
    if dtype_a != dtype_b:
        reason = f"dtype of x1 and x2 must be same, actual is {dtype_a}, {dtype_b}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)
    input_range = _get_input_range(range_x1, format_a,
                                   range_x2, format_b,
                                   range_bias, trans_a, trans_b, op_type)

    # check bias if bias in not None
    if bias:
        dtype_bias = bias.get("dtype")
        para_check.check_dtype_rule(dtype_bias, ("float16", "float32"), "bias")

    return dtype_a, dtype_out, input_range, shape_input


def fuzzy_range_check(input_x1: dict, input_x2: dict, bias: dict, trans_a: bool, trans_b: bool) -> bool:
    """
    check range for fuzzy compile
    """
    # check format and dtype
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    valid = (format_a == format_b and input_x1.get("dtype").lower() == input_x2.get("dtype").lower())
    if not valid:
        return False

    range_bias = None
    if bias is not None:
        bais_dtype = bias.get("dtype").lower()
        range_bias = bias.get("range")
        if bais_dtype not in ("float16", "float32"):
            return False
    op_type = "MatMul" if len(input_x1.get("ori_shape")) == 2 else "BatchMatMul"
    # check range in m,k,n
    err_msg = _get_input_range(range_x1, format_a, range_x2, format_b,
                                 range_bias, trans_a, trans_b, op_type, True)

    if err_msg == LOWER_LIMIT_STR:
        return False
    return True


def _get_var_name(format_a: str, format_b: str, cache_tiling_flag: bool) -> list:
    """
    Get the name of the variables
    """
    if format_a == "ND" and not cache_tiling_flag:
        m_var_name = "m_ori"
        k_var_name = "k_ori"
    else:
        m_var_name = "m"
        k_var_name = "k"

    if format_b == "ND" and not cache_tiling_flag:
        n_var_name = "n_ori"
    else:
        n_var_name = "n"
    if cache_tiling_flag:
        k_var_name = "k_ori"
    return [m_var_name, k_var_name, n_var_name]


def _set_shape(param_var: tuple, format_a: str, format_b: str, cache_tiling_flag: bool) -> tuple:
    """
    Set the shape for input
    """
    m_var, k_var, n_var = param_var
    if format_a == "ND" and cache_tiling_flag:
        m_var = m_var * BLOCK_CUBE
    if format_b == "ND" and cache_tiling_flag:
        n_var = n_var * BLOCK_CUBE
    return [k_var, m_var, n_var, k_var]


def _get_m_k_index(format_a: str, trans_a: bool) -> list:
    """
    get the correct m, k position for shape_x1.
    """
    if trans_a:
        m_index = -1 if format_a == "ND" else -2
        k_index = -2 if format_a == "ND" else -1
    else:
        m_index = -2 if format_a == "ND" else -1
        k_index = -1 if format_a == "ND" else -2
    return [m_index, k_index]


def _get_k_n_index(format_b: str, trans_b: bool) -> list:
    """
    get the correct k, n position for shape_x2.
    """
    if trans_b:
        n_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
        k_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
    else:
        n_index = -1 if format_b in ("ND", "FRACTAL_Z") else -2
        k_index = -2 if format_b in ("ND", "FRACTAL_Z") else -1
    return [k_index, n_index]


def _get_bias_tensor(bias: dict, format_b: str, n_var):
    """
    Get Bias Tensor
    """
    if bias:
        bias_dtype = bias.get("dtype")
        if format_b != "ND":
            bias_shape = [BLOCK_CUBE * n_var]
        else:
            bias_shape = [n_var]
        tensor_bias = tvm.placeholder(
            bias_shape, name="bias", dtype=bias_dtype, attrs={'ori_shape': bias_shape})
    else:
        tensor_bias = None
    return tensor_bias


def _get_real_trans(format_a: str, format_b: str, trans_a: bool, trans_b: bool) -> list:
    """
    Get the correct trans values used in compute
    """
    if format_a == Format.FRACTAL_NZ:
        trans_a = not trans_a
    if format_b == Format.FRACTAL_NZ:
        trans_b = not trans_b
    return [trans_a, trans_b]


def get_cache_tiling_flag(input_x1: dict, input_x2: dict, bias: dict) -> bool:
    """
    config if cache_tiling
    """
    support_l0c2out = tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out")
    if bias or support_l0c2out:
        return False
    if input_x1.get("range") and input_x2.get("range"):
        for dim_range_1, dim_range_2 in zip(input_x1.get("range"), input_x2.get("range")):
            input_none_range =  not dim_range_1 or None in dim_range_1 or not dim_range_2 or None in dim_range_2
            if input_none_range:
                return True
    else:
        return True
    return False


def _check_nd_in(input_x1: dict, input_x2: dict) -> bool:
    """
    check format of inputs
    support ND in ND out or ND in NZ out
    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    return format_a == "ND" and format_b == "ND"


def _define_cache_tiling_var(input_x1: dict, input_x2: dict, bias:dict) -> None:
    if get_cache_tiling_flag(input_x1, input_x2, bias):
        operation.var("k")
        operation.var("batch_single_core")
        operation.var("m_single_core")
        operation.var("n_single_core")
        operation.var("batch_dim")
        operation.var("n_dim")
        operation.var("m_dim")
        operation.var("k_dim")
        operation.var("m_al1")
        operation.var("n_bl1")
        operation.var("cub_n1")
        operation.var("m_l0")
        operation.var("k_l0")
        operation.var("n_ub_l0_time")
        operation.var("kal0_factor")
        operation.var("kbl0_factor")
        operation.var("kal1_factor")
        operation.var("kbl1_factor")
        operation.var("kal1_16")
        operation.var("kbl1_16")
        operation.var("kl1_times")
        if _check_nd_in(input_x1, input_x2):
            operation.var("m_aub")
            operation.var("n_bub")
            operation.var("k_aub")
            operation.var("k_bub")
            operation.var("multi_n_ub_l1")
            operation.var("multi_m_ub_l1")
            operation.var("multi_k_aub_l1")
            operation.var("multi_k_bub_l1")
            operation.var("a_align_value")
            operation.var("b_align_value")
            operation.var("aub_align_bound")
            operation.var("bub_align_bound")


def _check_empty_tensor(input_x1, input_x2, output_z, bias):
    # if tensor'shape has 0 or range's lower is 0
    if (check_dynamic_range_lower([input_x1, input_x2, output_z]) or
        is_empty_tensor_scene([input_x1, input_x2, output_z])):
        if bias:
            bias_range = bias.get("range")
            if bias_range and bias_range[0][0] == 0:
                bias["range"] = ((1, bias_range[0][1]), )
        check_tensor_shape({"tensor": [input_x1, input_x2, output_z],
                            "value": [-1, -1, -1],
                            "range": [(2, 2), (2, 2), (2, 2)]})


def batch_matmul_compute(input_x1: dict, input_x2: dict, bias: dict, offset_w: dict, output_z: dict, trans_a: bool,
                         trans_b: bool, offset_x: int, kernel_name: str, op_type: str = "BatchMatMulV2"):
    """
    batch_matmul computer

    Parameters:
    input_x1: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be fp16
    the format must be FRACTAL_NZ
    input_x2: dict
    A dict object, dict with keys(shape, dtype and range)
    the dtype must be fp16
    the format must be FRACTAL_NZ
    bias: dict
    A dict object, dict with keys(shape and format) or None
    the dtype must be fp16
    the format must be ND
    output_z: dict
    A dict object, dict with keys(shape and dtype)
    the dtype must be fp16
    the format must be FRACTAL_NZ
    trans_a: bool
    If true, shape_a == transposed before multiplication
    trans_b: bool
    If true, shape_a == transposed before multiplication
    kernel_name: str
    cce kernel_name
    Returns
    -------
    res : dict
    A dict object, dict with input tensor and output tensor
    """
    _check_empty_tensor(input_x1, input_x2, output_z, bias)
    # check soc_version
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "Hi3796CV300ES and Hi3796CV300CS and SD3403 don't support dynamic shape"
        )

    dtype_in, dtype_out, input_range, shape_input = check_and_config_para(
        input_x1, input_x2, bias, output_z, trans_a, trans_b, kernel_name, op_type
    )

    batch_range, m_range, k_range, n_range = input_range

    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    cache_tiling_flag = get_cache_tiling_flag(input_x1, input_x2, bias)
    m_var_name, k_var_name, n_var_name = _get_var_name(format_a, format_b, cache_tiling_flag)

    k_var = operation.var(k_var_name, k_range)
    m_var = operation.var(m_var_name, m_range)
    n_var = operation.var(n_var_name, n_range)
    shape_x1, shape_x2 = shape_input

    if op_type in ("BatchMatMulV2", "BatchMatMul"):
        batch_var = operation.var("batch", batch_range)
    elif op_type in ("MatMulV2", "MatMul"):
        shape_x1 = [DYNAMIC_FLAG, DYNAMIC_FLAG]
    else:
        reason = f"not support op_type: {op_type}"
        error_manager_vector.raise_err_specific_reson(op_type, reason)

    if soc_version in ("Ascend910", "Ascend310P"):
        _define_cache_tiling_var(input_x1, input_x2, bias)

    if len(shape_x1) >= BATCH_ND_LENGTH:
        shape_x1 = [batch_var, DYNAMIC_FLAG, DYNAMIC_FLAG]
    if len(shape_x2) >= BATCH_ND_LENGTH:
        shape_x2 = [batch_var, DYNAMIC_FLAG, DYNAMIC_FLAG]

    m_index, ka_index = _get_m_k_index(format_a, trans_a)
    kb_index, n_index = _get_k_n_index(format_b, trans_b)
    param_var = m_var, k_var, n_var
    shape_x1[ka_index], shape_x1[m_index], shape_x2[n_index], shape_x2[kb_index] = _set_shape(
        param_var, format_a, format_b, cache_tiling_flag)
    if format_a != "ND":
        shape_x1 = shape_x1 + [BLOCK_CUBE, BLOCK_CUBE]
    if format_b != "ND":
        shape_x2 = shape_x2 + [BLOCK_CUBE, BLOCK_CUBE]

    tensor_x1 = tvm.placeholder(shape_x1, name="tensor_a", dtype=dtype_in)
    tensor_x2 = tvm.placeholder(shape_x2, name="tensor_b", dtype=dtype_in)
    tensor_bias = _get_bias_tensor(bias, format_b, n_var)
    trans_a, trans_b = _get_real_trans(format_a, format_b, trans_a, trans_b)

    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": format_a,
        "format_b": format_b,
        "format_out": output_z.get("format"),
        "dst_dtype": dtype_out,
        "tensor_c": tensor_bias,
        "cache_tiling_flag": cache_tiling_flag,
        "kernel_name": kernel_name,
        "input_range": input_range
    }
    op_res = tbe.gemm(tensor_x1, tensor_x2, para_dict)

    tensor_list = [tensor_x1, tensor_x2]
    if bias:
        tensor_list.append(tensor_bias)
    return {"op_placeholder": tensor_list, "op_res": [op_res]}


@tbe_register.register_param_generalization("BatchMatMulV2")
def batch_matmul_v2_generalization(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                                   trans_a=False, trans_b=False, offset_x=0, kernel_name="batch_matmul",
                                   generalize_config: dict = None) -> list:
    """
    batch_matmul_v2_generalization

    Parameters:
    input_x1: A dict object, dict with keys(shape, dtype and range).
    the dtype must be fp16, the format must be FRACTAL_NZ
    input_x2: A dict object, dict with keys(shape, dtype and range)
    the dtype must be fp16, the format must be FRACTAL_NZ
    bias: A dict object, dict with keys(shape and format) or None
    the dtype must be fp16, the format must be ND
    output_z: A dict object, dict with keys(shape and dtype)
    the dtype must be fp16, the format must be FRACTAL_NZ
    trans_a: If true, shape_a == transposed before multiplication
    trans_b: If true, shape_a == transposed before multiplication
    kernel_name: cce kernel_name
    generalize_config: generalize config
    Returns
    -------
    res : A list object
    """
    # fuzzy compile
    if generalize_config.get("mode") == "keep_rank":
        result = []
        is_graph_mode = (util_gemm.is_graph_mode(input_x1) or util_gemm.is_graph_mode(input_x2))
        # single op mode or head node in graph
        input_len = 2 if bias is None else 3
        if not is_graph_mode:
            # get range generalization
            ori_range_x1 = util_gemm.cal_gemm_shape_range(input_x1["ori_shape"], input_x1["ori_format"])
            ori_range_x2 = util_gemm.cal_gemm_shape_range(input_x2["ori_shape"], input_x2["ori_format"])
            if ori_range_x1 == "LOWER_LIMIT" or ori_range_x2 == "LOWER_LIMIT":
                res = [{"result": "UNSUPPORTED", "reason": {"param_index": list(range(input_len)),
                                                            "type": ["lower_limit"] * input_len}}]
                return res

            util_gemm.generalize_input_keep_rank_gemm(input_x1)
            util_gemm.generalize_input_keep_rank_gemm(input_x2)
            input_x1["ori_range"], input_x2["ori_range"] = ori_range_x1, ori_range_x2
            if bias:
                ori_range_bias = util_gemm.cal_gemm_shape_range(bias["ori_shape"], bias["ori_format"])
                util_gemm.generalize_input_keep_rank_gemm(bias)
                bias["ori_range"] = ori_range_bias
            util_gemm.generalize_input_keep_rank_gemm(output_z)
            result.append([input_x1, input_x2, bias, offset_w, output_z,
                           {"trans_a": trans_a}, {"trans_b": trans_b}, {"offset_x": offset_x}])
        # graph mode
        else:
            status, err_msg = util_gemm.matmul_range_check(input_x1, input_x2, bias)
            # check result fail, return UNSUPPORTED json
            if not status:
                return err_msg
            status = fuzzy_range_check(input_x1, input_x2, bias, trans_a, trans_b)
            if not status:
                err_msg = [{"result": "UNSUPPORTED", "reason": {"param_index": list(range(input_len)),
                                                                "type": ["lower_limit"] * input_len}}]
                return err_msg
            # range check pass
            result.append([input_x1, input_x2, bias, offset_w, output_z,
                           {"trans_a": trans_a}, {"trans_b": trans_b}, {"offset_x": offset_x}])
        return result
    else:
        return


@register_operator("BatchMatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
    para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def batch_matmul_v2(input_x1, input_x2, bias=None, offset_w=None, output_z=None,
                    trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with nz format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    offset_w: dict
        A dict object, dict with keys(shape and dtype) or None
    output_z: dict
        A dict object, dict with keys(shape, dtype, format and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ or ND
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    offset_x: int
        offset of gradients in quant mode
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        None
    """

    with tbe.compute():
        res = batch_matmul_compute(
            input_x1, input_x2, bias, offset_w, output_z, trans_a, trans_b, offset_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }
    attr_cache_tiling = dict(res.get("op_res")[0].op.attrs.items()).get("cache_tiling", 0)
    if get_cache_tiling_flag(input_x1, input_x2, bias) and int(attr_cache_tiling) == 1:
        config.get("build_args")["predicate_realize_bound"] = False
        config.get("build_args")["enable_branch_eliminator_else_case"] = False
    tbe.build(sch, config)
    tbe_platform.fusion_manager.set_current_op_pattern("BatchMatmul")
