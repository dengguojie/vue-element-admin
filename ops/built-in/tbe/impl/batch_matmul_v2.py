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
batch_matmul
"""
import functools
from impl.dynamic.batch_matmul_v2 import base_op_select_format
from impl.dynamic.batch_matmul_v2 import gen_op_select_format_params

from impl import batch_matmul_vector
from impl.util import util_deconv_comm
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
DYNAMIC_UNRANK = [-2]
ND_LENGTH = 2
NZ_LENGTH = 4

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2

def _cal_min_l1space(dtype_b):
    block_reduce = tbe_platform.CUBE_MKN[dtype_b]["mac"][1]
    block_out = tbe_platform.CUBE_MKN[dtype_b]["mac"][2]
    mini_l1space = block_out * block_reduce * \
                   util_deconv_comm.BIT_RATIO_DICT.get(dtype_b)
    return mini_l1space


def get_op_support_info(input_x, # pylint: R0913,R0914,W0613
                        input_y,
                        bias=None,
                        offset_w=None,
                        output_z=None,
                        trans_a=False,
                        trans_b=False,
                        offset_x=0,
                        kernel_name="matmul"):
    """
    get the batch_matmul_v2 split, which only split batch, m and n, cannot cut k with bias

    """
    format_a = input_x.get("format")
    format_b = input_y.get("format")
    a_shape = input_x.get("shape")
    b_shape = input_y.get("shape")
    dtype_b = input_y.get("dtype")
    if format_a == 'FRACTAL_NZ':
        trans_a = not trans_a
    if format_b == 'FRACTAL_NZ':
        trans_b = not trans_b

    if format_a != 'FRACTAL_NZ':
        batch_len_a = len(a_shape) - ND_LENGTH
    else:
        batch_len_a = len(a_shape) - NZ_LENGTH
    if format_b in ('FRACTAL_NZ', 'FRACTAL_Z'):
        batch_len_b = len(b_shape) - NZ_LENGTH
    else:
        batch_len_b = len(b_shape) - ND_LENGTH

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
        batch_split_list = [[0, [i], [-1], [-1]]]
        if batch_len_b != 0:
            batch_split_list.append([1, [i], [-1], [-1]])
        axis_split_matrix_batch.append(
            [util_select_op_base.SplitInput(*batch_split_list),
                util_select_op_base.SplitOutput([0, [i]])]
        )

    out_m_axis = batch_len_a + 1 if format_a == "FRACTAL_NZ" else batch_len_a
    axis_split_matrix_a = [
        [util_select_op_base.SplitInput(m_split_list),
            util_select_op_base.SplitOutput([0, [out_m_axis]])]
    ]

    out_n_axis = batch_len_a if format_b in (
        "FRACTAL_NZ", "FRACTAL_Z") else batch_len_a + 1
    axis_split_matrix_b = [
        [util_select_op_base.SplitInput(*n_split_list),
            util_select_op_base.SplitOutput([0, [out_n_axis]])]
    ]

    axis_split_matrix = axis_split_matrix_a + \
        axis_split_matrix_b + axis_split_matrix_batch
    min_l1space = _cal_min_l1space(dtype_b)
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


# pylint: disable=locally-disabled,too-many-arguments,unnecessary-comprehension
# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b):
    """
    Check the given shape for matrix A, B and bias == legal

    Parameters
    ---------
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
            If True, shape_a is transposed before multiplication
    trans_b: bool
            If True, shape_b is transposed before multiplication

    Returns
    -------
    None
    """
    shape_len_a = len(shape_a)
    shape_len_b = len(shape_b)

    if shape_len_a >= shape_len_b:
        shape_len = shape_len_a
    else:
        shape_len = shape_len_b
    inp_src_dtype = src_dtype.lower()
    if inp_src_dtype == "float16":
        k_block_size = tbe_platform.BLOCK_REDUCE
    elif inp_src_dtype == "int8":
        k_block_size = tbe_platform.BLOCK_REDUCE_INT8

    check_list = ("float16", "int8")
    if inp_src_dtype not in check_list:
        error_manager_vector.raise_err_dtype_invalid('batch_matmul', 'input_x', check_list, inp_src_dtype)

    if shape_len < 2:
        error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                           "shape length for batch matmul greater than or equal to 2")

    is_gevm = (shape_a[-2] == 1) or (shape_a[-1] == 1)
    is_gemv = (shape_b[-2] == 1) or (shape_b[-1] == 1)

    if trans_a:
        m_shape = shape_a[shape_len_a - 1]
        km_shape = shape_a[shape_len_a - 2]
    else:
        m_shape = shape_a[shape_len_a - 2]
        km_shape = shape_a[shape_len_a - 1]

    if trans_b:
        kn_shape = shape_b[shape_len_b - 1]
        n_shape = shape_b[shape_len_b - 2]
    else:
        kn_shape = shape_b[shape_len_b - 2]
        n_shape = shape_b[shape_len_b - 1]

    if m_shape == 1:
        if n_shape == 1:
            error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input_x',
                                                               "input shape M and N can't both be 1")

    if km_shape != kn_shape:
        error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                           "reduce axis not same")

    shape_bias_length = len(shape_bias)

    if shape_bias_length > 0:
        if shape_bias_length == 1:
            if is_gevm or is_gemv:
                if shape_bias[0] != m_shape * n_shape:
                    error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                                       "broadcast case shape bias for "
                                                                       "gemv must be equal m*n")
            else:
                if shape_bias[0] != n_shape:
                    error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                                       "broadcast bias shape must be equal to shape n")
        elif shape_bias_length == shape_len:
            out_shape = [i for i in shape_a[:-2]] + [m_shape, n_shape]
            if [i for i in shape_bias] != out_shape:
                error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                                   "non broadcast bias shape "
                                                                   "must be same as output shape")
        else:
            error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                               "unsupport input shape now for batch bias case")


def _get_bias(shape_bias):
    """ get bias
    :param shape_bias
    :return: shape bias
    """
    bias_length = shape_bias[0]
    if bias_length % 16 != 0:
        bias_length = (bias_length // 16) * 16 + 16
        shape_bias = []
        shape_bias.append(bias_length)

    return shape_bias


def _get_input_shape(shape_x, transpose, src_dtype):
    """ get input shape
    :param shape_x
    :return:  input shape
    """
    shape_length = len(shape_x)
    dim_a = shape_x[shape_length - 2]
    dim_b = shape_x[shape_length - 1]
    shape_length = shape_length - 2
    res = shape_x[:shape_length]
    factor_base = 32 if src_dtype == "int8" else 16
    if transpose:
        factor_1 = factor_base
        factor_2 = 16
    else:
        factor_1 = 16
        factor_2 = factor_base
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


def _get_input_shape_b(shape_y, transpose, src_dtype):
    """ get input shape
    :param shape_x
    :return:  input shape
    """
    shape_length = len(shape_y)
    dim_a = shape_y[shape_length - 2]
    dim_b = shape_y[shape_length - 1]
    shape_length = shape_length - 2
    res = shape_y[:shape_length]
    factor_base = 32 if src_dtype == "int8" else 16
    if transpose:
        factor_1 = 16
        factor_2 = factor_base
    else:
        factor_1 = factor_base
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


# pylint: disable=too-many-return-statements
def _check_batch_range(input_x, input_y):
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
    if list(shape_a) == DYNAMIC_UNRANK or list(shape_b) == DYNAMIC_UNRANK:
        return True

    range_x1 = input_x.get("range")
    range_x2 = input_y.get("range")
    if not range_x1 or len(shape_a) <= ND_LENGTH:
        return False
    if not range_x2 or len(shape_b) < ND_LENGTH:
        return False

    batch_range_x1 = range_x1[:(len(shape_a) - ND_LENGTH)]
    batch_range_x2 = range_x2[:(len(shape_b) - ND_LENGTH)]

    if not batch_range_x2:
        return True

    if len(batch_range_x1) != len(batch_range_x2):
        return False

    for range1, range2 in zip(batch_range_x1, batch_range_x2):
        if range1[1] is not None and range2[1] is not None:
            if max(range1[0], range2[0]) > min(range1[1], range2[1]):
                return False

    return True


def op_select_format(input_x, input_y, bias=None, offset_w=None, output_z=None, trans_a=False,
                     trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    provide static format to FE
    """
    src_dtype = input_x.get("dtype")
    src_fp16_flag = True if src_dtype == "float16" else False
    _, full_case_senario_combinations = base_op_select_format(src_fp16_flag)

    param_list = gen_op_select_format_params(full_case_senario_combinations, is_batch_matmul_v2=True)
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def check_supported(input_x,
                    input_y,
                    bias=None,
                    offset_w={},
                    output_z={},
                    trans_a=False,
                    trans_b=False,
                    offset_x=0,
                    kernel_name="matmul"):
    """
    get the op supported situation
    """

    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    src_dtype = input_x.get("dtype")

    if any(v == 0 for v in shape_a) or any(v == 0 for v in shape_b):
        reason = "cannot support dim 0, shape_a:%s, shape_b:%s" \
                 % (str(shape_a), str(shape_b))
        return False, reason

    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x")
        para_check.check_shape(shape_b, param_name="input_y")
    else:
        if not _check_batch_range(input_x, input_y):
            reason = "The batch range or shape of inputs is illegal"
            return False, reason

    src_dtypes = ["float32", "int32"]
    if src_dtype in src_dtypes and not dynamic_flag:
        shape_length = len(shape_a)
        shape_length_b = len(shape_b)
        if shape_length != shape_length_b:
            reason = "The dimensions of TensorA and TensorB are not equal"
            return False, reason
        elif trans_b:
            if shape_b[shape_length - 2] == 1:
                reason = "When trans_b is True and src_dtypes is float32 or int32," \
                    "the shape_b[{}] cannot be equal to 1".format(shape_length - 2)
                return False, reason
        elif bool(1 - trans_b):
            if shape_b[shape_length - 1] == 1:
                reason = "When trans_b is False and src_dtypes is float32 or int32," \
                    "the shape_b[{}] cannot be equal to 1".format(shape_length - 1)
                return False, reason
        elif trans_a:
            if trans_b:
                if shape_a[shape_length - 2] != shape_b[shape_length - 1]:
                    reason = "The shape_a[{}] and shape_b[{}] are not equal".format(
                        shape_length - 2, shape_length - 1)
                    return False, reason
            else:
                if shape_a[shape_length - 2] != shape_b[shape_length - 2]:
                    reason = "The shape_a[{0}] and shape_b[{0}] are not equal".format(shape_length - 2)
                    return False, reason
        else:
            if trans_b:
                if shape_a[shape_length - 1] != shape_b[shape_length - 1]:
                    reason = "The shape_a[{0}] and shape_b[{0}] are not equal".format(shape_length - 1)
                    return False, reason
            else:
                if shape_a[shape_length - 1] != shape_b[shape_length - 2]:
                    reason = "The shape_a[{0}] and shape_b[{1}] are not equal".format(
                        shape_length - 1, shape_length - 2)
                    return False, reason
    elif src_dtype in ["float16", "int8"] and not dynamic_flag:
        shape_length = len(shape_a)
        if trans_a:
            k_shape = shape_a[shape_length - 2]
        else:
            k_shape = shape_a[shape_length - 1]

        shape_length_b = len(shape_b)
        if trans_b:
            k_b_shape = shape_b[shape_length_b - 1]
        else:
            k_b_shape = shape_b[shape_length_b - 2]

        if k_shape != k_b_shape:
            reason = "The K of TensorA and TensorB should be equal," \
                "but actually K of TensorA is {} and K of TensorB is {}".format(k_shape, k_b_shape)
            return False, reason

    return True, ""


# pylint: disable=simplifiable-if-expression,unexpected-keyword-arg,no-value-for-parameter
@tbe_platform.fusion_manager.register("batch_matmul_v2")
def batch_matmul_compute(input_x, input_y, bias=None, offset_w={}, output_z={}, trans_a=False,
                         trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_z: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    format_a = input_x.op.attrs["format"].value
    format_b = input_y.op.attrs["format"].value
    if format_a == 'FRACTAL_NZ':
        trans_a_local = not trans_a
    else:
        trans_a_local = trans_a

    if format_b == 'FRACTAL_NZ':
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b
    dst_dtype = output_z.get("dtype").lower()

    ori_shape_x = input_x.op.attrs["ori_shape"]
    ori_shape_y = input_y.op.attrs["ori_shape"]
    ori_shape_out = output_z.get("ori_shape")
    batch_shape_a = ori_shape_x[:-2] if len(ori_shape_x) > 2 else list()
    batch_shape_b = ori_shape_y[:-2] if len(ori_shape_y) > 2 else list()
    batch_shape_out = ori_shape_out[:-2] if len(ori_shape_out) > 2 else list()

    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("batch_matmul",
                                                      "For BatchMatMulV2, tensor offset_w must be None!")
    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "dst_dtype": dst_dtype,
        "tensor_c": bias,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "kernel_name": kernel_name,
        "batch_shape_a": batch_shape_a,
        "batch_shape_b": batch_shape_b,
        "batch_shape_out": batch_shape_out
    }
    result = tbe.gemm(tensor_a=input_x, tensor_b=input_y, para_dict=para_dict)

    return result

# pylint: disable=simplifiable-if-expression,unexpected-keyword-arg,no-value-for-parameter
def batch_matmul_compute_self(input_x, input_y, bias=None,  offset_w={}, output_z={}, trans_a=False,
                              trans_b=False, offset_x=0, kernel_name="matmul"):
    """
    algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_z: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    cube_vector_split = tbe_platform.get_soc_spec("CUBE_VECTOR_SPLIT")
    format_a = input_x.op.attrs["format"].value
    format_b = input_y.op.attrs["format"].value
    if format_a == 'FRACTAL_NZ' and not cube_vector_split:
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a

    if format_b == 'FRACTAL_NZ' and not cube_vector_split:
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b
    dst_dtype = output_z.get("dtype").lower()

    batch_shape_a = input_x.op.attrs["ori_batch_shape"]
    batch_shape_b = input_y.op.attrs["ori_batch_shape"]
    ori_shape_out = output_z.get("ori_shape")
    batch_shape_out = ori_shape_out[:-2] if len(ori_shape_out) > 2 else list()
    if offset_w is not None:
        error_manager_vector.raise_err_specific_reson("batch_matmul_v2",
                                                      "For BatchMatMulV2, tensor offset_w must be None!")
    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "dst_dtype": dst_dtype,
        "offset_a": offset_x,
        "offset_b": offset_w,
        "tensor_c": bias,
        "kernel_name": kernel_name,
        "batch_shape_out": batch_shape_out,
        "batch_shape_a": batch_shape_a,
        "batch_shape_b": batch_shape_b
        }
    result = tbe.gemm(tensor_a=input_x, tensor_b=input_y, para_dict=para_dict)

    return result


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals, no-member
# pylint: disable=too-many-statements, dangerous-default-value
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_BOOL, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def batch_matmul_v2(input_x, input_y, bias=None, offset_w={}, output_z={}, trans_a=False,
                    trans_b=False, offset_x=0, kernel_name="matmul"):
    """ algorithm: batch_matmul
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters
    ---------
    input_x: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    input_y: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    bias: dict
        A dict object, contanis a 1-dimensional tensor's info:
        the shape and type and format, the type can be float16,
        float32, int32, the shape must be 1-dimensional,
        the format can be [ND, NHWC]
    output_z: dict
        A dict object, contains a matrix's type and
        shape and format, the type can be float16,
        float32, int32, the length of shape must be
        greater than 2, the format can be [ND, NHWC, FRACTAL_NZ]
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: str
        If true, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "matmul"

    Return
    ------
    None
    """
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")
    shape_a_length = len(shape_a)
    shape_b_length = len(shape_b)
    if shape_a is not None:
        if shape_a_length < 2:
            shape_a = list(shape_a)
            shape_a.insert(0, 1)
    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = list(shape_b)
            shape_b.append(1)
    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        if input_x.get("format") == "FRACTAL_NZ":
            shape_bias = _get_bias(shape_bias)

    src_dtype = input_x.get("dtype").lower()
    dst_dtype = output_z.get("dtype").lower()
    is_fractal = False

    shape_a = list(shape_a)
    shape_b = list(shape_b)
    if input_x.get("format") == "FRACTAL_NZ":
        shape_a = _get_input_shape(shape_a, trans_a, src_dtype)
        shape_b = _get_input_shape_b(shape_b, trans_b, src_dtype)

    para_check.check_shape(shape_a, param_name="input_x")
    para_check.check_shape(shape_b, param_name="input_y")

    trans_a_local = trans_a
    trans_b_local = trans_b

    if input_x.get("format") == "FRACTAL_NZ":
        batch_axis = shape_a[:(len(shape_a) - 2)]
        shape_a = batch_axis + [shape_a[len(shape_a) - 1], shape_a[len(shape_a) - 2]]
        trans_a_local = bool(1 - trans_a)

    if input_y.get("format") == "FRACTAL_NZ":
        batch_axis = shape_b[:(len(shape_b) - 2)]
        shape_b = batch_axis + [shape_b[len(shape_b) - 1], shape_b[len(shape_b) - 2]]
        trans_b_local = bool(1 - trans_b)

    if src_dtype.lower() == "float32" or src_dtype.lower() == "int32":
        batch_matmul_vector.matmul_vector_cce(shape_a, shape_b, src_dtype, trans_a, trans_b, shape_bias, kernel_name)
        return

    _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a_local, trans_b_local)
    inp_src_dtype = src_dtype.lower()

    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_b) - 2]
    n_shape = shape_b[len(shape_b) - 1]

    if inp_src_dtype == "float16":
        block_reduce = tbe_platform.BLOCK_REDUCE
    elif inp_src_dtype == "int8":
        block_reduce = tbe_platform.BLOCK_REDUCE_INT8

    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT

    if trans_a and km_shape == 1:
        block_in = tbe_platform.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = tbe_platform.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out = tbe_platform.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = tbe_platform.BLOCK_VECTOR

    if trans_a:
        shape_a_dup = (m_shape // block_in, km_shape // block_reduce, block_reduce, block_in)
    else:
        shape_a_dup = (m_shape // block_reduce, km_shape // block_in, block_in, block_reduce)

    if trans_b:
        shape_b_dup = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_dup = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x.get("format") == "FRACTAL_Z":
        input_x["format"] = "FORMAT_FRACTAL_Z"
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "fractal"
    elif input_x.get("format") == "FRACTAL_NZ":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_dup = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_y.get("format") == "FRACTAL_Z":
        input_y["format"] = "FORMAT_FRACTAL_Z"
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "fractal"
    elif input_y.get("format") == "FRACTAL_NZ":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_dup = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    batch_shape_a = None
    ori_batch_shape_a = list()
    if len(shape_a) > 2:
        batch_shape_a = functools.reduce(lambda x, y: x * y, shape_a[:-2])
        ori_batch_shape_a = list(shape_a[:-2])

    batch_shape_b = None
    ori_batch_shape_b = list()
    if len(shape_b) > 2:
        batch_shape_b = functools.reduce(lambda x, y: x * y, shape_b[:-2])
        ori_batch_shape_b = list(shape_b[:-2])

    if len(shape_a) >= len(shape_b):
        batch_shape = batch_shape_a
    else:
        batch_shape = batch_shape_b

    if batch_shape is not None and batch_shape >= 1:
        if batch_shape_a is not None:
            shape_a_dup = (batch_shape_a,) + shape_a_dup
        if batch_shape_b is not None:
            shape_b_dup = (batch_shape_b,) + shape_b_dup

    tensor_bias = None
    shape_bias_length = len(shape_bias)
    if shape_bias_length <= 2:
        shape_bias_dup = shape_bias
    else:
        shape_bias_dup = (shape_bias[len(shape_bias) - 2], shape_bias[len(shape_bias) - 1])
        bias_batch_size = functools.reduce(lambda x, y: x * y, shape_bias[:-2])
        shape_bias_dup = (bias_batch_size,) + shape_bias_dup

    tensor_a = tvm.placeholder(shape_a_dup, name='tensor_a',
                               attrs={'format': format_a,
                                      'ori_batch_shape': ori_batch_shape_a,
                                      "ori_shape": input_x.get("ori_shape")},
                               dtype=inp_src_dtype)
    tensor_b = tvm.placeholder(shape_b_dup, name='tensor_b',
                               attrs={'format': format_b,
                                      'ori_batch_shape': ori_batch_shape_b,
                                      "ori_shape": input_y.get("ori_shape")},
                               dtype=inp_src_dtype)

    if shape_bias_length > 0:
        bias_dtype = bias.get("dtype")
        tensor_bias = tvm.placeholder(shape_bias_dup, name='tensor_bias',
                                      dtype=bias_dtype, attrs={'ori_shape': bias['ori_shape']})

    if offset_w is None:
        tensor_offset_w = None
    else:
        error_manager_vector.raise_err_specific_reson(
            "batch_matmul", "offset w must be None!")

    result = batch_matmul_compute_self(tensor_a, tensor_b, tensor_bias, tensor_offset_w,
                                       output_z, trans_a, trans_b, offset_x, kernel_name)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)
    tensor_list = [tensor_a, tensor_b, result]

    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.build(schedule, config)
    tbe_platform.fusion_manager.set_current_op_pattern("BatchMatmul")
