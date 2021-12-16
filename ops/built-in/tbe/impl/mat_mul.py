# Copyright 2019 Huawei Technologies Co., Ltd
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
matmul
"""
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.dynamic.mat_mul import base_op_select_format
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from tbe.common.context import op_context

from impl.matmul_vector import matmul_vector_cce

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2
_C0_16 = 16


def op_select_format(input_x, input_y, bias=None, offset_w=None, output_z=None, trans_a=False,
                     trans_b=False, offset_x=0, kernel_name="matmul", impl_mode=""):
    """
    provide dynamic format to FE
    """
    # BatchMatMulV1 does not support offset_w
    src_dtype = input_x.get("dtype")
    src_fp16_flag = True if src_dtype == "float16" else False
    _, full_case_senario_combinations = base_op_select_format(src_fp16_flag)

    param_list = gen_op_select_format_params(full_case_senario_combinations, support_offset_w=True)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    src_dtype: str
            The data type of input, support "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication

    Returns None
    """
    shape_len = len(shape_a)
    src_dtype = src_dtype.lower()

    if shape_len != len(shape_b):
        error_detail = "length of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("mat_mul", "x1",
                                                               "x2", error_detail)

    if shape_len != 2:
        error_detail = "length of shape must be 2, more than 2 dimensions should use batch_matmul now!"
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x1", error_detail)

    is_gevm = bool((shape_a[-2] == 1) or (shape_a[-1] == 1))
    is_gemv = bool((shape_b[-2] == 1) or (shape_b[-1] == 1))

    if trans_a:
        m_shape = shape_a[-1]
        km_shape = shape_a[-2]
    else:
        m_shape = shape_a[-2]
        km_shape = shape_a[-1]

    if trans_b:
        kn_shape = shape_b[-1]
        n_shape = shape_b[-2]
    else:
        kn_shape = shape_b[-2]
        n_shape = shape_b[-1]

    if m_shape == 1:
        if n_shape == 1:
            error_detail = "input shape x1 and x2 can't both be 1"
            error_manager_vector.raise_err_two_input_shape_invalid("mat_mul", "x1",
                                                                   "x2", error_detail)

    if km_shape != kn_shape:
        error_detail = "reduce axis of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("mat_mul", "x1",
                                                               "x2", error_detail)

    shape_bias_length = len(shape_bias)
    if shape_bias_length > 0:
        if shape_bias_length == 1:
            if is_gevm or is_gemv:
                if shape_bias[0] != m_shape * n_shape:
                    error_detail = "broadcast case shape bias for gemv must be equal m*n"
                    error_manager_vector.raise_err_input_shape_invalid(
                        "mat_mul", "bias", error_detail)
            else:
                if shape_bias[0] != n_shape:
                    error_detail = "broadcast bias shape must be equal to shape n"
                    error_manager_vector.raise_err_input_shape_invalid(
                        "mat_mul", "bias", error_detail)
        elif shape_bias_length == shape_len:
            if [i for i in shape_bias[-2:]] != [m_shape, n_shape]:  # pylint: disable=unnecessary-comprehension
                error_detail = "non broadcast bias shape must be same as output shape"
                error_manager_vector.raise_err_input_shape_invalid(
                    "mat_mul", "bias", error_detail)
        else:
            error_detail = "unsupport input shape now for batch bias case"
            error_manager_vector.raise_err_input_shape_invalid(
                "mat_mul", "bias", error_detail)


def _shape_check_quantification(shape_a, shape_b, trans_a, trans_b, format_a):
    if trans_a:
        m_shape = shape_a[1]
        k_shape = shape_a[0]
    else:
        m_shape = shape_a[0]
        k_shape = shape_a[1]

    if trans_b:
        n_shape = shape_b[0]
    else:
        n_shape = shape_b[1]

    if k_shape % 32 != 0:
        error_detail = "the K value must be multiple of 32!"
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x1", error_detail)

    if format_a == "FORMAT_FRACTAL_Z":
        if m_shape % 16 != 0 and m_shape != 1:
            error_detail = "the M value must be 1 or multiple of 16 when the format is FORMAT_FRACTAL_Z!"
            error_manager_vector.raise_err_input_shape_invalid(
                "mat_mul", "x1", error_detail)

    if n_shape % 16 != 0:
        error_detail = "the K value must be multiple of 16!"
        error_manager_vector.raise_err_input_shape_invalid(
            "mat_mul", "x2", error_detail)


def _get_bias(shape_bias, ori_shape_bias):
    bias_length = 1
    for i in shape_bias:
        bias_length *= i
    ori_bias_length = 1
    for i in ori_shape_bias:
        ori_bias_length *= i
    return [(bias_length + _C0_16 - 1) // _C0_16 * _C0_16], [ori_bias_length]


def _align_shape_a(shape_x, transpose):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = []
    if transpose:
        factor_1 = 32
        factor_2 = 16
    else:
        factor_1 = 16
        factor_2 = 32

    res.append((dim_a + factor_1 - 1) // factor_1 * factor_1)
    res.append((dim_b + factor_2 - 1) // factor_2 * factor_2)
    return res


def _align_shape_b(shape_y, transpose):
    dim_a = shape_y[0]
    dim_b = shape_y[1]
    res = []
    if transpose:
        factor_1 = 16
        factor_2 = 32
    else:
        factor_1 = 32
        factor_2 = 16

    res.append((dim_a + factor_1 - 1) // factor_1 * factor_1)
    res.append((dim_b + factor_2 - 1) // factor_2 * factor_2)
    return res


def _cal_min_l1space(dtype_b):
    block_reduce = tbe_platform.CUBE_MKN[dtype_b]["mac"][1]
    block_out = tbe_platform.CUBE_MKN[dtype_b]["mac"][2]
    mini_l1space = block_out * block_reduce * \
                   util_deconv_comm.BIT_RATIO_DICT.get(dtype_b)
    return mini_l1space


def get_op_support_info(input_x1,  # pylint: R0913,R0914,W0613
                        input_x2,
                        bias,
                        offset_w={},
                        output_y={},
                        trans_a=False,
                        trans_b=False,
                        offset_x=0,
                        kernel_name="matmul",
                        impl_mode=""):
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

    # input/output Serial， axis Serial, （split enable(0:enable, -1：unable), split enable）
    # cut m
    if not trans_a:
        m_split_list = [0, [0], [0], [0]]
        mk_split_list = [0, [1]]
    else:
        m_split_list = [0, [1], [0], [0]]
        mk_split_list = [0, [0]]
    # cut n
    if not trans_b:
        n_split_list = [[1, [1], [0], [0]]]
        nk_split_list = [1, [0]]
    else:
        n_split_list = [[1, [0], [0], [0]]]
        nk_split_list = [1, [1]]

    if bias:
        axis_reduce_list = None
        n_split_list.append([2, [0], [0], [0]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                            util_select_op_base.ReduceOutput([0, 1, False])]]

    if format_a == "FRACTAL_NZ":
        axis_split_matrix_a = [
            [util_select_op_base.SplitInput(m_split_list),
             util_select_op_base.SplitOutput([0, [1]])],
        ]
    else:
        axis_split_matrix_a = [
            [util_select_op_base.SplitInput(m_split_list),
             util_select_op_base.SplitOutput([0, [0]])],
        ]

    # cut n
    if format_b in ("FRACTAL_NZ", "FRACTAL_Z"):
        axis_split_matrix_b = [
            [util_select_op_base.SplitInput(*n_split_list),
             util_select_op_base.SplitOutput([0, [0]])],
        ]
    else:
        axis_split_matrix_b = [
            [util_select_op_base.SplitInput(*n_split_list),
             util_select_op_base.SplitOutput([0, [1]])],
        ]

    axis_split_matrix = axis_split_matrix_a + axis_split_matrix_b
    min_l1space = _cal_min_l1space(dtype_b)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def _is_fuzzily_build():
    """
    check fuzzily build flag
    """
    context = op_context.get_context()
    return (context and context.get_build_type() == "fuzzily_build")

def check_supported(input_x1,
                    input_x2,
                    bias,
                    offset_w={},
                    output_y={},
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="matmul",
                    impl_mode=""):
    """
    check the op support situation
    """
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    src_dtype = input_x1.get("dtype")
    cube_type = ["float16", "int8", "int4"]
    if (format_a == "FRACTAL_NZ" or format_b == "FRACTAL_NZ") and \
            src_dtype in cube_type:
        return True, ""
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")

    if any(v == 0 for v in shape_a) or any(v == 0 for v in shape_b):
        reason = "cannot support dim 0, shape_a:%s, shape_b:%s" \
                 % (str(shape_a), str(shape_b))
        return False, reason

    src_dtype = input_x1.get("dtype")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x1")
        para_check.check_shape(shape_b, param_name="input_x2")
    target_type = ["float32", "int32"]
    res = True, ""
    if src_dtype in target_type and not dynamic_flag:
        if len(shape_a) != 2 and len(shape_b) != 2:
            reason = "the input_shape is not supported, shape_a:%s, shape_b%s"\
                     % (str(shape_a), str(shape_b))
            res = False, reason
        elif trans_a:
            if trans_b:
                if shape_a[0] != shape_b[1]:
                    reason = "the input_shape is not equal, shape_a[0]:%s, shape_b[1]:%s"\
                             % (shape_a[0], shape_b[1])
                    res = False, reason
            elif shape_a[0] != shape_b[0]:
                reason = "the input_shape is not equal, shape_a[0]:%s, shape_b[0]:%s"\
                         % (shape_a[0], shape_b[0])
                res = False, reason
        elif trans_b:
            if shape_a[1] != shape_b[1]:
                reason = "the input_shape is not equal, shape_a[1]:%s, shape_b[1]:%s"\
                         % (shape_a[1], shape_b[1])
                res = False, reason
        elif shape_a[1] != shape_b[0]:
            reason = "the input_shape is not equal, shape_a[1]:%s, shape_b[0]:%s"\
                     % (shape_a[1], shape_b[0])
            res = False, reason
    elif src_dtype in cube_type and not dynamic_flag:
        if len(shape_a) != 2 and len(shape_b) != 2:
            reason = "the input_shape is not supported, shape_a:%s, shape_b%s"\
                     % (str(shape_a), str(shape_b))
            res = False, reason
        if trans_a:
            k_shape = shape_a[0]
        else:
            k_shape = shape_a[1]

        if trans_b:
            k_b_shape = shape_b[1]
        else:
            k_b_shape = shape_b[0]

        if k_shape != k_b_shape:
            reason = "the shape not equal, k_shape:%s, k_b_shape:%s" % (k_shape, k_b_shape)
            res = False, reason
    if _is_fuzzily_build() and src_dtype != "float16":
        reason = "in dynamic mode, src dtype only support float16!"
        res = False, reason

    return res


@tbe_platform.fusion_manager.register("mat_mul")
def mat_mul_compute(input_x1,
                    input_x2,
                    bias,
                    offset_w={},
                    output_y={},
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="matmul",
                    impl_mode=""):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Returns
    -------
    None
    """
    format_a = input_x1.op.attrs["format"].value
    format_b = input_x2.op.attrs["format"].value
    if format_a == 'FRACTAL_NZ':
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a

    if format_b == 'FRACTAL_NZ':
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b

    dst_dtype = output_y.get("dtype").lower()

    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("mat_mul",
                                                      "For MatMul, tensor offset_w must be None!")

    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name,
        "impl_mode": impl_mode,
        "format_out": output_y.get("format"),
    }
    result = tbe.gemm(tensor_a=input_x1, tensor_b=input_x2,
                      para_dict=para_dict)

    return result

def mat_mul_compute_self(input_x1,
                         input_x2,
                         bias,
                         offset_w={},
                         output_y={},
                         trans_a=False,
                         trans_b=False,
                         offset_x=0,
                         kernel_name="matmul",
                         impl_mode=""):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A matrix(2D Tensor), the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Returns
    -------
    None
    """
    format_a = input_x1.op.attrs["format"].value
    format_b = input_x2.op.attrs["format"].value
    if format_a == 'FRACTAL_NZ':
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a

    if format_b == 'FRACTAL_NZ':
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b

    dst_dtype = output_y.get("dtype").lower()

    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("mat_mul",
                                                      "For MatMul, tensor offset_w must be None!")

    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name,
        "impl_mode": impl_mode,
        "format_out": output_y.get("format"),
        }
    result = tbe.gemm(tensor_a=input_x1, tensor_b=input_x2, para_dict=para_dict)

    return result


def _matmul_vector_one_compute(tensor_a, tensor_b, tensor_bias, axis):
    """
    algorithm: _matmul_vector_one
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    tensor_a: TVM tensor
        The dtype support "float32", "int32".
    tensor_b: TVM tensor
        The dtype support "float32", "int32".
    tensor_bias: TVM tensor
        The dtype support "float32", "int32".
    axis：int
        the axis for reduce.

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as tensor_a.
    """
    dtype = tensor_a.dtype
    shape_a = shape_util.shape_to_list(tensor_a.shape)
    shape_b = shape_util.shape_to_list(tensor_b.shape)
    if tensor_bias is not None:
        shape_bias = shape_util.shape_to_list(tensor_bias.shape)

    shape_a, shape_b, shape_max = \
        shape_util.broadcast_shapes(shape_a, shape_b, param_name_input1="tensor_a",
                                    param_name_input2="tensor_b")
    tensor_b = tbe.broadcast(tensor_b, shape_max, dtype)
    res_tmp = tbe.vmul(tensor_a, tensor_b)
    res = tbe.reduce_sum(res_tmp, axis=axis)

    shape_res = shape_util.shape_to_list(res.shape)
    if tensor_bias is not None:
        shape_res, shape_bias, shape_max2 = \
            shape_util.broadcast_shapes(shape_res, shape_bias, param_name_input1="res",
                                        param_name_input2="bias")
        tensor_bias = tbe.broadcast(tensor_bias, shape_max2, dtype)
        res = tbe.vadd(res, tensor_bias)

    return res


def _matmul_vector_one(shape_a, shape_b, src_type, trans_a, trans_b, bias, kernel_name="matmul_vector"):
    """
    algorithm: _matmul_vector_one
    calculating  matrix multiplication with bias, use vector mode ,C = A*B + bias

    Parameters
    ----------
    shape_a : list or tuple
        shape of tensor_a
    shape_b : list or tuple
        shape of tensor_b
    src_type : str
        the data type, assume src_dtype equals dst_dtype, support float32 and int32
    trans_a : bool
        if the tensor A need transport, the value == True
    trans_b : bool
        if the tensor B need transport, the value == True
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    kernel_name : str
        cce kernel name, default value == "matmul_vector"

    Returns
    -------
    None
    """
    axis = 0 if trans_a else 1
    if (trans_a and trans_b) or (not trans_a and not trans_b):
        shape_b = (shape_b[1], shape_b[0])

    tensor_a = tvm.placeholder(shape_a, name='tensor_a', dtype=src_type)
    tensor_b = tvm.placeholder(shape_b, name='tensor_b', dtype=src_type)
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)

    shape_bias_length = len(shape_bias)
    tensor_bias = None
    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(
            shape_bias, name='tensor_bias', dtype=src_type)

    result = _matmul_vector_one_compute(tensor_a, tensor_b, tensor_bias, axis)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, result]
    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedule, config)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def mat_mul(input_x1,
            input_x2,
            bias,
            offset_w={},
            output_y={},
            trans_a=False,
            trans_b=False,
            offset_x=0,
            kernel_name="matmul",
            impl_mode=""):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    input_x2: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be float16,
        float32, int32, the origin shape must be 2-dimensional,
        the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Returns
    -------
    None
    """
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)

    # if ori_shape is invalid use shape
    if shape_a is not None and shape_a_length < 2:
        shape_a = input_x1.get("shape")

    # if ori_shape is invalid use shape
    if shape_b is not None and shape_b_length < 2:
        shape_b = input_x2.get("shape")

    shape_a = list(shape_a)
    shape_b = list(shape_b)

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = _align_shape_a(shape_a, trans_a)
        shape_b = _align_shape_b(shape_b, trans_b)

    para_check.check_shape(shape_a, param_name="input_x1")
    para_check.check_shape(shape_b, param_name="input_x2")

    align_shape_bias = ()
    ori_shape_bias = ()
    if bias is not None and bool(bias):
        align_shape_bias, ori_shape_bias = _get_bias(bias.get("shape"), bias.get("ori_shape"))

    src_dtype = input_x1.get("dtype").lower()

    target_type = ["float32", "int32"]
    if src_dtype in target_type and not tbe_platform.get_soc_spec("CUBE_VECTOR_SPLIT"):
        if (trans_b and shape_b[0] == 1) or (not trans_b and shape_b[1] == 1):
            _matmul_vector_one(shape_a, shape_b, src_dtype, trans_a, trans_b,
                               bias, kernel_name)
        else:
            matmul_vector_cce(shape_a, shape_b, src_dtype, trans_a, trans_b,
                              align_shape_bias, kernel_name)

        return

    if src_dtype not in ("int8", "int4"):
        if input_x1.get("format") == "FRACTAL_NZ":
            _shape_check(shape_a, shape_b, align_shape_bias, src_dtype, trans_a, trans_b)
        else:
            _shape_check(shape_a, shape_b, ori_shape_bias, src_dtype, trans_a, trans_b)
    else:
        _shape_check_quantification(shape_a, shape_b, trans_a, trans_b,
                                    input_x1.get("format"))

    shape_a_temp = input_x1.get("shape")
    shape_b_temp = input_x2.get("shape")
    tensor_bias = None
    format_a = input_x1.get("format")
    format_b = input_x2.get("format")
    tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                               attrs={'format': format_a,
                                      'ori_shape': input_x1.get("ori_shape")},
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                               attrs={'format': format_b,
                                      'ori_shape': input_x2.get("ori_shape")},
                               dtype=src_dtype)
    shape_bias_length = len(align_shape_bias)
    if shape_bias_length > 0:
        bias_dtype = bias.get("dtype")
        tensor_bias = tvm.placeholder(align_shape_bias, name='tensor_bias',
                                      dtype=bias_dtype, attrs={'ori_shape': ori_shape_bias})

    if offset_w is None:
        tensor_offset_w = None
    else:
        error_manager_vector.raise_err_specific_reson(
            "mat_mul", "offset_w must be None!")

    result = mat_mul_compute_self(tensor_a, tensor_b, tensor_bias, tensor_offset_w,
                                  output_y,
                                  trans_a, trans_b, offset_x, kernel_name, impl_mode)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, result]
    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.build(schedule, config)
