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
matmulcompress
"""
from impl.util import util_select_op_base
from impl.util.util_common import cal_mini_l1_size_matmul
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.utils.error_manager import error_manager_vector

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
L1FUSION_INPUT_CTR = 2

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
        error_manager_vector.raise_err_input_shape_invalid(
            "compress_mat_mul", "x1", "the K value must be multiple of 32!")

    if format_a == "FORMAT_FRACTAL_Z":
        if m_shape % 16 != 0 and m_shape != 1:
            error_manager_vector.raise_err_input_shape_invalid("compress_mat_mul", "x1",
                "the M value must be 1 or multiple of 16 when the format is FORMAT_FRACTAL_Z!")

    if n_shape % 16 != 0:
        error_manager_vector.raise_err_input_shape_invalid(
            "compress_mat_mul", "x2", "the K value must be multiple of 16!")


def _get_bias(shape_bias):
    bias_length = shape_bias[0]
    if bias_length % 16 != 0:
        bias_length = (bias_length // 16) * 16 + 16
        shape_bias = []
        shape_bias.append(bias_length)

    return shape_bias


def _get_input_shape(shape_x, transpose):
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = []
    if transpose:
        factor_1 = 32
        factor_2 = 16
    else:
        factor_1 = 16
        factor_2 = 32

    if dim_a % factor_1 != 0:
        dim_a = (dim_a // factor_1) * factor_1 + factor_1
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % factor_2 != 0:
        dim_b = (dim_b // factor_2) * factor_2 + factor_2
        res.append(dim_b)
    else:
        res.append(dim_b)
    return res


def _get_input_shape_b(shape_y, transpose):
    dim_a = shape_y[0]
    dim_b = shape_y[1]
    res = []
    if transpose:
        factor_1 = 16
        factor_2 = 32
    else:
        factor_1 = 32
        factor_2 = 16

    if dim_a % factor_1 != 0:
        dim_a = (dim_a // factor_1) * factor_1 + factor_1
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % factor_2 != 0:
        dim_b = (dim_b // factor_2) * factor_2 + factor_2
        res.append(dim_b)
    else:
        res.append(dim_b)
    return res

# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=unused-argument, too-many-statements
# pylint: disable=dangerous-default-value
def check_supported(input_x1,
                    input_x2,
                    compress_index,
                    bias,
                    offset_w={},
                    output_y={},
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="compress_matmul"):
    """
    check the op support situation
    """
    cube_type = ["float16", "int8"]
    if (input_x1.get("format") == "FRACTAL_NZ" or input_x2.get("format") == "FRACTAL_NZ") and \
            input_x1.get("dtype") in cube_type:
        return True
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x1")
        para_check.check_shape(shape_b, param_name="input_x2")
    target_type = ["float32", "int32"]
    res = True
    if input_x1.get("dtype") in target_type and not dynamic_flag:
        if len(shape_a) != 2 and len(shape_b) != 2:
            res = False
        elif trans_a:
            if trans_b:
                if shape_a[0] != shape_b[1]:
                    res = False
            elif shape_a[0] != shape_b[0]:
                res = False
        elif trans_b:
            if shape_a[1] != shape_b[1]:
                res = False
        elif shape_a[1] != shape_b[0]:
            res = False
    elif input_x1.get("dtype") in cube_type:
        if len(shape_a) != 2 and len(shape_b) != 2:
            res = False
        if trans_a:
            k_shape = shape_a[0]
        else:
            k_shape = shape_a[1]

        if trans_b:
            k_b_shape = shape_b[1]
        else:
            k_b_shape = shape_b[0]

        if not dynamic_flag and k_shape != k_b_shape:
            res = False

    return res


# pylint: disable=locally-disabled, too-many-arguments,unexpected-keyword-arg,no-value-for-parameter
# pylint: disable=locally-disabled, simplifiable-if-expression
# pylint: disable=too-many-locals, too-many-statements, dangerous-default-value
@tbe_platform.fusion_manager.fusion_manager.register("compress_mat_mul")
def compress_mat_mul_compute(input_x1,
                             input_x2,
                             compress_index,
                             bias,
                             offset_w={},
                             output_y={},
                             trans_a=False,
                             trans_b=False,
                             offset_x=0,
                             kernel_name="compress_matmul"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_NZ]
    input_x2: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_Z]
    compress_index: the dict of input compress index
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type must be [int32, float16],
        the shape must be 1-dimensional, the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be [float16, int32], the
        shape must be 2-dimensional, the format can be [FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "compress_matmul"

    Returns
    -------
    None
    """
    if input_x1.op.attrs["format"].value == 'FRACTAL_NZ':
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a

    if input_x2.op.attrs["format"].value == 'FRACTAL_NZ':
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b

    dst_dtype = output_y.get("dtype").lower()


    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("compress_mat_mul",
                                                      "For Compress MatMul, tensor offset_w must be None!")

    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": input_x1.op.attrs["format"].value,
        "format_b": input_x2.op.attrs["format"].value,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "compress_index": compress_index,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name
        }

    return tbe.gemm(tensor_a=input_x1, tensor_b=input_x2, para_dict=para_dict)

# pylint: disable=simplifiable-if-expression,unexpected-keyword-arg,no-value-for-parameter
def compress_mat_mul_compute_self(input_x1,
                                  input_x2,
                                  compress_index,
                                  bias,
                                  offset_w={},
                                  output_y={},
                                  trans_a=False,
                                  trans_b=False,
                                  offset_x=0,
                                  kernel_name="compress_matmul"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_NZ]
    input_x2: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_Z]
    compress_index: the dict of input compress index
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type must be [int32, float16],
        the shape must be 1-dimensional, the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be [float16, int32], the
        shape must be 2-dimensional, the format can be [FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "compress_matmul"

    Returns
    -------
    None
    """
    cube_vector_split = tbe_platform.cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")
    format_a = input_x1.op.attrs["format"].value
    format_b = input_x2.op.attrs["format"].value
    if format_a == 'FRACTAL_NZ' and not cube_vector_split:
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a

    if format_b == 'FRACTAL_NZ' and not cube_vector_split:
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b

    dst_dtype = output_y.get("dtype").lower()

    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("compress_mat_mul",
                                                      "For Compress MatMul, tensor offset_w must be None!")

    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "compress_index": compress_index,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name
    }
    result = tbe.gemm(tensor_a=input_x1, tensor_b=input_x2, para_dict=para_dict)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def get_op_support_info(input_x1,
                        input_x2,
                        compress_index,
                        bias=None,
                        offset_w=None,
                        output_y=None,
                        trans_a=False,
                        trans_b=False,
                        offset_x=0,
                        kernel_name="compress_matmul"):
    """
    get the matmul split, which only split the m and n, cannot cut k with bias

    """
    dtype_b = input_x2.get("dtype")
    trans_a = not trans_a

    # input/output Serial, axis Serial, (0: overlap -1: without overlap)
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
        n_split_list.append([3, [0], [-1], [-1]])
    else:
        # cut k_dim which is reduce dim
        axis_reduce_list = [[util_select_op_base.ReduceInput(mk_split_list, nk_split_list),
                            util_select_op_base.ReduceOutput([0, "REDUCE_ADD", False])]]

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


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals, too-many-statements, dangerous-default-value
# pylint: disable=too-many-locals, line-too-long
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_BOOL, para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def compress_mat_mul(input_x1,
                     input_x2,
                     compress_index,
                     bias,
                     offset_w={},
                     output_y={},
                     trans_a=False,
                     trans_b=False,
                     offset_x=0,
                     kernel_name="compress_matmul"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    input_x1: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_NZ]
    input_x2: dict
        A dict object, contain a matrix(2D Tensor) 's type and
        shape and format, the type can be int8, the shape must
        be 2-dimensional, the format can be [FRACTAL_Z]
    compress_index: the dict of input compress index
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type must be [int32, float16],
        the shape must be 1-dimensional, the format can be [ND, NHWC]
    output_y: dict
        A dict object, contains a matrix(2D Tensor) 's type and
        shape and format, the type can be [float16, int32], the
        shape must be 2-dimensional, the format can be [FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "compress_matmul"

    Returns
    -------
    None
    """
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)

    if shape_a is not None:
        if shape_a_length < 2:
            shape_a = input_x1.get("shape")

    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = input_x2.get("shape")

    shape_a = list(shape_a)
    shape_b = list(shape_b)

    para_check.check_format(input_x1.get("format"), ('FRACTAL_NZ'), param_name="input_x1")
    shape_a = _get_input_shape(shape_a, trans_a)
    shape_b = _get_input_shape_b(shape_b, trans_b)

    para_check.check_shape(shape_a, param_name="input_x1")
    para_check.check_shape(shape_b, param_name="input_x2")

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        shape_bias = _get_bias(shape_bias)

    src_dtype = input_x1.get("dtype").lower()
    para_check.check_dtype(src_dtype, ['int8'], param_name="input_x1")
    dst_dtype = output_y.get("dtype").lower()

    _shape_check_quantification(shape_a, shape_b, trans_a, trans_b,
                                    input_x1.get("format"))

    shape_a_temp = input_x1.get("shape")
    shape_b_temp = input_x2.get("shape")
    tensor_bias = None
    format_a = "FRACTAL_NZ"
    format_b = "FRACTAL_Z"

    tensor_a = tvm.placeholder(shape_a_temp, name='tensor_a',
                               attrs={'format': format_a},
                               dtype=src_dtype)
    tensor_b = tvm.placeholder(shape_b_temp, name='tensor_b',
                               attrs={'format': format_b},
                               dtype=src_dtype)
    index_size = tvm.var("index_size", dtype="int32")
    compress_index = tvm.placeholder([index_size, ],
                                     name='compress_index', dtype="int8")
    shape_bias_length = len(shape_bias)
    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(shape_bias, name='tensor_bias',
                                      dtype=dst_dtype)

    if offset_w is None:
        tensor_offset_w = None
    else:
        error_manager_vector.raise_err_specific_reson(
            "compress_mat_mul", "offset_w must be None!")

    result = compress_mat_mul_compute_self(tensor_a, tensor_b, compress_index, tensor_bias, tensor_offset_w,
                                  output_y, trans_a, trans_b, offset_x, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)

    tensor_list = [tensor_a, tensor_b, compress_index, result]
    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, compress_index, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.cce_build_code(schedule, config)
