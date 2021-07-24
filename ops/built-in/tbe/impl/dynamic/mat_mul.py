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
dynamic mat_mul
"""
import math

from impl.util import fusion_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_common import cal_mini_l1_size_matmul

# General limitation of the size for input shape: 2**32 - 1
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
MKN_MIN = 1
MKN_MAX = 2147483648
BLOCK_CUBE = 16
DYNAMIC_FLAG = -1
DYNAMIC_FLAG_UNRANK = [-2]
NZ_LENGTH = 4
ND_LENGTH = 2
L1FUSION_INPUT_CTR = 2



def get_op_support_info(input_x1, input_x2, bias, offset_w=None, output_y=None,
                        trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    dtype_b = input_x2.get("dtype")
    if format_a == 'FRACTAL_NZ':
        trans_a = not trans_a
    if format_b == 'FRACTAL_NZ':
        trans_b = not trans_b

    # cut m
    if not trans_a:
        m_split_list = [0, [0], [-1], [-1]]
        mk_split_list = [0, [1]]
    else:
        m_split_list = [0, [1], [-1], [-1]]
        mk_split_list = [0, [0]]
    # cut n
    if not trans_b:
        n_split_list = [[1, [1], [-1], [-1]]]
        nk_split_list = [1, [0]]
    else:
        n_split_list = [[1, [0], [-1], [-1]]]
        nk_split_list = [1, [1]]

    if bias:
        axis_reduce_list = None
        n_split_list.append([2, [0], [-1], [-1]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                            util_select_op_base.ReduceOutput([0, 1, False])]]

    # cut m
    axis_split_matrix_a = [
        [util_select_op_base.SplitInput(m_split_list),
            util_select_op_base.SplitOutput([0, [1]])],
    ]
    # cut n
    axis_split_matrix_b = [
        [util_select_op_base.SplitInput(*n_split_list),
            util_select_op_base.SplitOutput([0, [0]])],
    ]

    axis_split_matrix = axis_split_matrix_a + axis_split_matrix_b
    min_l1space = cal_mini_l1_size_matmul(dtype_b)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def _check_format(real_format, expect_format, param_name):
    """
    check format
    """
    if real_format != expect_format:
        error_manager_vector.raise_err_input_format_invalid(
            "mat_mul", param_name, expect_format, real_format)


def _get_range_intersection(range1, range2, param_name):
    """
    get range intersection of two range
    """
    if range1[1] is None:
        return range2
    if range2[1] is None:
        return range1

    range_ins = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    if range_ins[0] > range_ins[1]:
        reson = "interval is not valid of " + param_name
        error_manager_vector.raise_err_specific_reson("mat_mul", reson)
    return range_ins


def _get_input_range(range_x1, range_x2, range_bias, trans_a, trans_b):
    """
    get range in m, k, n
    """
    if range_x1 and len(range_x1) == NZ_LENGTH:
        if trans_a:
            m_range = list(range_x1[0])
            k_range_x1 = list(range_x1[1])
        else:
            k_range_x1 = list(range_x1[0])
            m_range = list(range_x1[1])
    else:
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "length of x1_range must be 4"
        )

    if range_x2 and len(range_x2) == NZ_LENGTH:
        if trans_b:
            k_range_x2 = list(range_x2[0])
            n_range = list(range_x2[1])
        else:
            n_range = list(range_x2[0])
            k_range_x2 = list(range_x2[1])
    else:
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "length of x2_range must be 4"
        )

    k_range = _get_range_intersection(k_range_x1, k_range_x2, "k_range")
    if range_bias:
        range_bias_n = list(range_bias[0])
        if range_bias[0][1] is not None:
            range_bias_n = [math.ceil(i / BLOCK_CUBE) for i in range_bias[0]]
        n_range = _get_range_intersection(n_range, range_bias_n, "n_range")

    return [m_range, k_range, n_range]


def _check_dynamic_mode(shape_x1, shape_x2):
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

    if all([i != DYNAMIC_FLAG for i in shape_x1]) and all([i != DYNAMIC_FLAG for i in shape_x2]):
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "dynamic must at least one in m,k,n"
        )


def _get_dynamic_shape_and_range(input_x1, input_x2, bias):
    """
    get the shape and range of matmul
    """
    shape_x1 = input_x1.get("ori_shape")
    shape_x2 = input_x2.get("ori_shape")
    range_x1 = input_x1.get("range")
    range_x2 = input_x2.get("range")
    bias_range = None

    if list(shape_x1) == DYNAMIC_FLAG_UNRANK:
        shape_x1 = (-1, -1)
        range_x1 = ((1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))
    if list(shape_x2) == DYNAMIC_FLAG_UNRANK:
        shape_x2 = (-1, -1)
        range_x2 = ((1, None), (1, None), (BLOCK_CUBE, BLOCK_CUBE), (BLOCK_CUBE, BLOCK_CUBE))

    if bias:
        bias_range = bias.get("range")

    return [shape_x1, shape_x2], [range_x1, range_x2, bias_range]


def check_and_config_para(input_x1, input_x2, bias, output_y,
                          trans_a, trans_b, kernel_name):
    """
    check and config dynamic mode
    """

    # get format and dtype
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    format_out = output_y.get("format")
    dtype_a = input_x1.get("dtype").lower()
    dtype_b = input_x2.get("dtype").lower()
    dtype_out = output_y.get("dtype").lower()

    # check kernel_name dtype and format
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype_rule(dtype_a, ["float16"], "x1")
    para_check.check_dtype_rule(dtype_b, ["float16"], "x2")
    para_check.check_dtype_rule(dtype_out, ["float16"], "output")
    _check_format(format_a, "FRACTAL_NZ", "x1")
    _check_format(format_b, "FRACTAL_NZ", "x2")
    _check_format(format_out, "FRACTAL_NZ", "output")

    # get range and ori_shape
    shape_input, range_input = _get_dynamic_shape_and_range(input_x1, input_x2, bias)
    range_x1, range_x2, range_bias = range_input
    shape_x1, shape_x2 = shape_input

    # check dynamic mode
    _check_dynamic_mode(shape_x1, shape_x2)
    # get range in m,k,n
    input_range = _get_input_range(range_x1, range_x2, range_bias, trans_a, trans_b)

    # check bias if bias in not None
    if bias:
        dtype_bias = bias.get("dtype")
        para_check.check_dtype_rule(dtype_bias, ("float16", "float32"), "bias")

    return dtype_a, dtype_out, input_range


def _mat_mul_compute(input_x1, input_x2, bias, offset_w, output_y, trans_a, trans_b, offset_x, kernel_name):
    """
    matmul computer

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
    offset_w: None
        input offset_w tensor
    output_y: dict
        A dict object, dict with keys(shape and dtype)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    trans_a: bool
        If true, shape_a == transposed before multiplication
    trans_b: bool
        If true, shape_a == transposed before multiplication
    offset_x:
        offset of x
    kernel_name: str
        cce kernel_name
    Returns
    -------
    res : dict
        A dict object, dict with input tensor and output tensor
    """

    # check offset
    if offset_w:
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", 'offset_w must be None!'
        )
    if offset_x != 0:
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", 'offset_x must be 0!'
        )

    # check soc_version
    soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
    if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "Hi3796CV300ES and Hi3796CV300CS don't support dynamic shape"
        )

    dtype_in, dtype_out, input_range = check_and_config_para(
        input_x1, input_x2, bias, output_y, trans_a, trans_b, kernel_name
    )

    m_range, k_range, n_range = input_range

    shape_x1_nz = [DYNAMIC_FLAG, DYNAMIC_FLAG, BLOCK_CUBE, BLOCK_CUBE]
    shape_x2_nz = [DYNAMIC_FLAG, DYNAMIC_FLAG, BLOCK_CUBE, BLOCK_CUBE]

    m_var = operation.var("m", m_range)
    k_var = operation.var("k", k_range)
    n_var = operation.var("n", n_range)

    # only support NZ for dynamic mode
    trans_a = not trans_a
    trans_b = not trans_b
    if not trans_a:
        shape_x1_nz[0] = m_var
        shape_x1_nz[1] = k_var
    else:
        shape_x1_nz[0] = k_var
        shape_x1_nz[1] = m_var

    if not trans_b:
        shape_x2_nz[0] = k_var
        shape_x2_nz[1] = n_var
    else:
        shape_x2_nz[0] = n_var
        shape_x2_nz[1] = k_var

    tensor_x1 = tvm.placeholder(shape_x1_nz, name="tensor_a", dtype=dtype_in)
    tensor_x2 = tvm.placeholder(shape_x2_nz, name="tensor_b", dtype=dtype_in)
    if bias:
        bias_dtype = bias.get("dtype")
        bias_shape = [BLOCK_CUBE * n_var]
        tensor_bias = tvm.placeholder(
            bias_shape, name="bias", dtype=bias_dtype)
    else:
        tensor_bias = None
    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": "FRACTAL_NZ",
        "format_b": "FRACTAL_NZ",
        "dst_dtype": dtype_out,
        "tensor_c": tensor_bias,
        "kernel_name": kernel_name
    }
    op_res = tbe.gemm(tensor_x1, tensor_x2, para_dict)
    tensor_list = [tensor_x1, tensor_x2]
    if bias:
        tensor_list.append(tensor_bias)
    return {"op_placeholder": tensor_list, "op_res": [op_res]}


@register_operator_compute("MatMul", op_mode="dynamic", support_fusion=False)
def mat_mul_fuse_compute(input_x1, input_x2, bias, offset_w, output_y,
                         trans_a=False, trans_b=False, offset_x=0,
                         kernel_name="matmul"):
    """
    matmul computer for fusion

    Parameters:
    input_x1: tensor
    input_x2: tensor
    bias: tensor or None
    offset_w: None
    output_y: dict
        A dict object, dict with keys(shape, dtype, format and range)
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
    fusion_util.check_fusion_input([input_x1])
    fusion_util.check_fusion_input([input_x2])
    if bias:
        fusion_util.check_fusion_input([bias])

    # set fusion build config
    build_cfg = tbe_register.get_fusion_buildcfg()
    build_cfg['constant_realize_extent_in_infer_bound'] = False

    para_dict = {
        "trans_a": trans_a,
        "trans_b": trans_b,
        "format_a": "FRACTAL_NZ",
        "format_b": "FRACTAL_NZ",
        "tensor_c": bias,
        "kernel_name": kernel_name
    }
    op_res = tbe.gemm(input_x1, input_x2, para_dict)
    tensor_list = [input_x1, input_x2]
    if bias:
        tensor_list.append(bias)
    return {"op_placeholder": tensor_list, "op_res": [op_res]}


def _generate_unknown_shape(shape):
    return [DYNAMIC_FLAG for i in shape]


def _generalize_input_keep_rank(input_dict):
    if input_dict["format"] in ("NHWC", "ND"):
        input_dict["shape"] = _generate_unknown_shape(input_dict["shape"])
        input_dict["ori_shape"] = _generate_unknown_shape(input_dict["ori_shape"])
    else:
        x_old_1 = input_dict["shape"][-1]
        x_old_2 = input_dict["shape"][-2]
        input_dict["shape"] = _generate_unknown_shape(input_dict["shape"])
        input_dict["ori_shape"] = _generate_unknown_shape(input_dict["ori_shape"])
        input_dict["shape"][-1] = x_old_1
        input_dict["shape"][-2] = x_old_2


@tbe_register.register_param_generalization("MatMul")
def  matmul_generalization(input_x1, input_x2, bias, offset_w={}, output_y={},
                           trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul",
                           generalize_config={"mode": "keep_rank"}):
    result = []
    if generalize_config["mode"] == "keep_rank": #fuzzy compile
        _generalize_input_keep_rank(input_x1)
        _generalize_input_keep_rank(input_x2)
        if bias:
            _generalize_input_keep_rank(bias)
        _generalize_input_keep_rank(output_y)
        result.append([input_x1, input_x2, bias, offset_w, output_y,
                       {"trans_a": trans_a}, {"trans_b": trans_b}, {"offset_x": offset_x}])
    else:
        error_manager_cube.raise_err_one_para(
            "E62306",
            "MatMul",
            "Invalid generalize mode, currently only support keep_rank"
        )

    return result


@register_operator("MatMul")
@register_operator("MatMulV2")
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
    para_check.REQUIRED_ATTR_BOOL,
    para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def mat_mul(input_x1, input_x2, bias, offset_w={}, output_y={},
            trans_a=False, trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    caculating matrix multiplication with bias, C = A * B + bias
    only support input with nz format and fp16 in dynamic mode

    Parameters:
    input_x1: dict
        A dict object, dict with keys(shape, dtype, and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    input_x2: dict
        A dict object, dict with keys(shape, dtype and range)
        the dtype must be fp16
        the format must be FRACTAL_NZ
    bias: dict
        A dict object, dict with keys(shape and dtype) or None
        the dtype must be fp16
        the format must be ND
    offset_w: None
        input offset_w tensor
    output_y: dict
        A dict object, dict with keys(shape, dtype, format and range)
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
        None
    """
    with tbe.compute():
        res = _mat_mul_compute(input_x1, input_x2, bias, offset_w, output_y,
                               trans_a, trans_b, offset_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res.get("op_res"))

    tensor_list = res.get("op_placeholder") + res.get("op_res")
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extent_in_infer_bound": False}
    }
    tbe.build(sch, config)
