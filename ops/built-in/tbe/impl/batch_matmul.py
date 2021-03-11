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
# pylint: disable=ungrouped-imports
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from tbe import tvm
from te.utils import para_check
from impl import batch_matmul_vector
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base


# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
DYNAMIC_UNRANK = [-2]
ND_LENGTH = 2


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
    k_block_size = tbe_platform.BLOCK_REDUCE
    check_list = ("float16")

    if inp_src_dtype not in check_list:
        error_manager_vector.raise_err_dtype_invalid('batch_matmul', 'input_x', check_list, inp_src_dtype)

    if shape_len < 2:
        error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                           "shape length for batch matmul greater than or equal to 2")

    if len(shape_a) == len(shape_b):
        if shape_a[:shape_len_a - 2] != shape_b[:shape_len_b - 2]:
            error_manager_vector.raise_err_two_input_shape_invalid('batch_matmul', 'input_x', 'input_y',
                                                                   "batch size of a and b are not equal")

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

    if m_shape % tbe_platform.BLOCK_IN != 0 and m_shape != 1:
        error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                           "input shape M should be 1 or multiple of %d" %
                                                           tbe_platform.cce_params.BLOCK_IN)
    if m_shape != 1:
        if km_shape % k_block_size != 0:
            error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                               "input shape K1 should be multiple of %d" %
                                                               tbe_platform.cce_params.BLOCK_IN)

    if n_shape % tbe_platform.BLOCK_IN != 0 and n_shape != 1:
        error_manager_vector.raise_err_input_shape_invalid('batch_matmul', 'input',
                                                           "input shape N should be 1 or multiple of %d" %
                                                           tbe_platform.cce_params.BLOCK_IN)

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


def _get_input_shape(shape_x):
    """ get input shape
    :param shape_x
    :return:  input shape
    """
    shape_length = len(shape_x)
    dim_a = shape_x[shape_length - 2]
    dim_b = shape_x[shape_length - 1]
    shape_length = shape_length - 2
    res = shape_x[:shape_length]
    if dim_a % 16 != 0:
        dim_a = (dim_a // 16) * 16 + 16
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % 16 != 0:
        dim_b = (dim_b // 16) * 16 + 16
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

# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=dangerous-default-value, no-member
# pylint: disable=too-many-statements, unused-argument
def op_select_format(input_x, input_y, bias=None, output_z={}, trans_a=False,
                     trans_b=False, kernel_name="matmul"):
    """
    provide dynamic format to FE
    """
    src_dtype = input_x.get("dtype")
    shape_a = input_x.get("ori_shape")
    shape_b = input_y.get("ori_shape")

    is_dynamic_shape = any(v == -1 for v in shape_a) or any(v == -1 for v in shape_b)
    if is_dynamic_shape:
        input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                               datatype="float16",
                                               format="FRACTAL_NZ",
                                               unknownshape_format="FRACTAL_NZ")
        input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                               datatype="float16",
                                               format="FRACTAL_NZ",
                                               unknownshape_format="FRACTAL_NZ")
        input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                               datatype="float16",
                                               format="ND",
                                               unknownshape_format="ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16",
                                                format="FRACTAL_NZ",
                                                unknownshape_format="FRACTAL_NZ")
    else:
        if src_dtype == "float16":
            input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                                   datatype="float16,float16",
                                                   format="FRACTAL_NZ,FRACTAL_NZ")
            input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                                   datatype="float16,float16",
                                                   format="FRACTAL_NZ,FRACTAL_NZ")
            input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                                   datatype="float16,float",
                                                   format="ND,ND")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float",
                                                    format="FRACTAL_NZ,FRACTAL_NZ")
        else:
            input0 = util_select_op_base.gen_param(classify="input0", name="x1",
                                                   datatype="float16,float,float,int32,int32",
                                                   format="FRACTAL_NZ,NHWC,ND,NHWC,ND")
            input1 = util_select_op_base.gen_param(classify="input1", name="x2",
                                                   datatype="float16,float,float,int32,int32",
                                                   format="FRACTAL_NZ,NHWC,ND,NHWC,ND")
            input2 = util_select_op_base.gen_param(classify="input2", name="bias",
                                                   datatype="float16,float,float,int32,int32",
                                                   format="ND,NHWC,ND,NHWC,ND")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float,float,int32,int32",
                                                    format="FRACTAL_NZ,NHWC,ND,NHWC,ND")

    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-arguments, no-member,no-else-return
# pylint: disable=too-many-statements, unused-argument,too-many-return-statements
def check_supported(input_x, input_y, bias=None, output_z={}, trans_a=False,
                    trans_b=False, kernel_name="matmul"):
    """
    get the op supported situation
    """

    shape_a = input_x.get("shape")
    shape_b = input_y.get("shape")
    src_dtype = input_x.get("dtype")
    dynamic_flag = any(v < 0 for v in shape_a) or any(v < 0 for v in shape_b)
    if not dynamic_flag:
        para_check.check_shape(shape_a, param_name="input_x")
        para_check.check_shape(shape_b, param_name="input_y")
    else:
        if not _check_batch_range(input_x, input_y):
            return False

    src_dtypes = ["float32", "int32"]
    if src_dtype in src_dtypes and not dynamic_flag:
        shape_length = len(shape_a)
        shape_length_b = len(shape_b)
        if shape_length != shape_length_b:
            return False
        elif trans_b:
            if shape_b[shape_length - 2] == 1:
                return False
        elif bool(1-trans_b):
            if shape_b[shape_length - 1] == 1:
                return False
        elif trans_a:
            if trans_b:
                if shape_a[shape_length - 2] != shape_b[shape_length - 1]:
                    return False
            else:
                if shape_a[shape_length - 2] != shape_b[shape_length - 2]:
                    return False
        else:
            if trans_b:
                if shape_a[shape_length - 1] != shape_b[shape_length - 1]:
                    return False
            else:
                if shape_a[shape_length - 1] != shape_b[shape_length - 2]:
                    return False
    elif src_dtype == "float16" and not dynamic_flag:
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
            return False

    return True

# pylint: disable=simplifiable-if-expression,unexpected-keyword-arg,no-value-for-parameter
@tbe_platform.fusion_manager.fusion_manager.register("batch_matmul")
def batch_matmul_compute(input_x, input_y, bias=None, output_z={}, trans_a=False,
                         trans_b=False, kernel_name="matmul"):
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
    batch_shape = ori_shape_x[:-2] if len(ori_shape_x) > 2 else None


    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "kernel_name": kernel_name,
        "batch_shape": batch_shape
        }
    result = tbe.gemm(tensor_a=input_x, tensor_b=input_y, para_dict=para_dict)
    return result

# pylint: disable=simplifiable-if-expression,unexpected-keyword-arg,no-value-for-parameter
def batch_matmul_compute_self(input_x, input_y, bias=None, output_z={}, trans_a=False,
                              trans_b=False, kernel_name="matmul"):
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
        trans_a_local = False if trans_a else True
    else:
        trans_a_local = trans_a


    if format_b == 'FRACTAL_NZ':
        trans_b_local = False if trans_b else True
    else:
        trans_b_local = trans_b
    dst_dtype = output_z.get("dtype").lower()

    para_dict = {
        "trans_a": trans_a_local,
        "trans_b": trans_b_local,
        "format_a": format_a,
        "format_b": format_b,
        "tensor_c": bias,
        "dst_dtype": dst_dtype,
        "kernel_name": kernel_name
        }
    result = tbe.gemm(tensor_a=input_x, tensor_b=input_y, para_dict=para_dict)

    return result


# pylint: disable=locally-disabled,too-many-arguments
# pylint: disable=too-many-locals, no-member
# pylint: disable=too-many-statements, dangerous-default-value
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_BOOL, para_check.KERNEL_NAME)
def batch_matmul(input_x, input_y, bias=None, output_z={}, trans_a=False,
                 trans_b=False, kernel_name="matmul"):
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
            shape_a = input_x.get("shape")

    if shape_b is not None:
        if shape_b_length < 2:
            shape_b = input_y.get("shape")
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
        shape_a = _get_input_shape(shape_a)
        shape_b = _get_input_shape(shape_b)

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
        shape_a_dup = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_dup = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_dup = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_dup = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)

    if input_x.get("format") == "FORMAT_FRACTAL_Z":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "fractal"
    elif input_x.get("format") == "FRACTAL_NZ":
        shape_a_dup = (shape_a_dup[0], shape_a_dup[1], shape_a_dup[2], shape_a_dup[3])
        format_a = "FRACTAL_NZ"
    else:
        shape_a_dup = (shape_a[len(shape_a) - 2], shape_a[len(shape_a) - 1])
        format_a = "ND"

    if input_y.get("format") == "FORMAT_FRACTAL_Z":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "fractal"
    elif input_y.get("format") == "FRACTAL_NZ":
        shape_b_dup = (shape_b_dup[0], shape_b_dup[1], shape_b_dup[2], shape_b_dup[3])
        format_b = "FRACTAL_NZ"
    else:
        shape_b_dup = (shape_b[len(shape_b) - 2], shape_b[len(shape_b) - 1])
        format_b = "ND"

    batch_shape_a = None
    if len(shape_a) > 2:
        batch_shape_a = functools.reduce(lambda x, y: x * y, shape_a[:-2])

    batch_shape_b = None
    if len(shape_b) > 2:
        batch_shape_b = functools.reduce(lambda x, y: x * y, shape_b[:-2])

    if len(shape_a) >= len(shape_b):
        batch_shape = batch_shape_a
    else:
        batch_shape = batch_shape_b

    if batch_shape is not None and batch_shape >= 1:
        if is_fractal:
            if batch_shape_a is not None:
                shape_a_dup = (batch_shape_a,) + shape_a_dup
            if batch_shape_b is not None:
                shape_b_dup = (batch_shape_b,) + shape_b_dup
        else:
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
                               attrs={'format': format_a},
                               dtype=inp_src_dtype)
    tensor_b = tvm.placeholder(shape_b_dup, name='tensor_b',
                               attrs={'format': format_b},
                               dtype=inp_src_dtype)

    if shape_bias_length > 0:
        tensor_bias = tvm.placeholder(shape_bias_dup, name='tensor_bias',
                                      dtype=dst_dtype)
    result = batch_matmul_compute_self(tensor_a, tensor_b, tensor_bias,
                                       output_z, trans_a, trans_b, kernel_name)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(result)
    tensor_list = [tensor_a, tensor_b, result]

    if shape_bias_length > 0:
        tensor_list = [tensor_a, tensor_b, tensor_bias, result]

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.cce_build_code(schedule, config)
    tbe_platform.fusion_manager.fusion_manager.set_current_op_pattern("BatchMatmul")
