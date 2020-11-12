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
dynamic greater_equal
"""
import te.lang.cce as tbe
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm


# define a scalar, value = 2**(-126), minimun num of float32 2**(-126)
SCALAR_MIN_FP32 = 2 ** (-126)
# define a scalar, value = 2**(62)
SCALAR_MUL_FP32 = 2 ** (62)
# define a scalar, value = 2**(2)
SCALAR_MUL1_FP32 = 2 ** (2)
# define a scalar, value = 2**(-24), minimun num of float16 2**(-24)
SCALAR_MIN_FP16 = 2 ** (-24)
# define a scalar, value = 2**(12)
SCALAR_MUL_FP16 = 2 ** (12)
# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = 0
SCALAR_ZERO = 0


# 'pylint: disable=unused-argument,invalid-name
def _greater_equal_compare(data, shape, dtype, data_min):
    """
    greater equal compare.

    Parameters:
    ----------
    data : tuple, two input data
    shape : list or tuple, shape of input data
    dtype : str
        source data type, support float16,float32,int32,int8,uint8
    data_min : tvm.const
        the minimal data according to dtype

    Returns
    -------
    the compare result
    """
    if dtype == "int32":
        data_one = tbe.broadcast(tvm.const(SCALAR_ONE, "float16"), shape, "float16")
    else:
        data_one = tbe.broadcast(tvm.const(SCALAR_ONE, dtype), shape, dtype)

    res_sub = tbe.vsub(data[1], data[0])
    res_min = tbe.vmins(res_sub, data_min)
    res_max = tbe.vmaxs(res_min, tvm.const(SCALAR_ZERO, dtype))

    if dtype == "float32":
        # max num of float32 is 2**126
        # but cce can only support 2**62, so use 62/62/2 to adaptor 126
        res_mul1 = tbe.vmuls(res_max, tvm.const(SCALAR_MUL_FP32, dtype=dtype))
        res_mul2 = tbe.vmuls(res_mul1, tvm.const(SCALAR_MUL_FP32, dtype=dtype))
        res_mul = tbe.vmuls(res_mul2, tvm.const(SCALAR_MUL1_FP32, dtype=dtype))
    elif dtype == "float16":
        # max num of float16 is 2**24
        # but cce can only support 2**12, so use 12/12 to adaptor 24
        res_mul1 = tbe.vmuls(res_max, tvm.const(SCALAR_MUL_FP16, dtype=dtype))
        res_mul = tbe.vmuls(res_mul1, tvm.const(SCALAR_MUL_FP16, dtype=dtype))
    else:
        res_mul = tbe.cast_to(res_max, "float16")
    res = tbe.vsub(data_one, res_mul)

    return tbe.cast_to(res, "uint8", True)


def greater_equal_compute(input_x, input_y, output_z, kernel_name="greater_equal"):
    """
    if x is greater than y or equals y, then return 1, else return 0.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x, has shape, dtype and range attributes
    input_y: TVM tensor
        the placeholder of input_y, has shape, dtype and range attributes
    output_z: dict
        dict info of output_z
    kernel_name: str
        cce kernel name, default value is "greater_equal"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)
    shape_x, shape_y, shape_max = \
        shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x", param_name_input2="input_y")

    dtype_x = input_x.dtype
    if dtype_x in ("int8", "uint8"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")
        dtype_x = "float16"

    input_x = tbe.broadcast(input_x, shape_max)
    input_y = tbe.broadcast(input_y, shape_max)

    if dtype_x == "float32":
        # minimun num of float32 2**(-126)
        data_min = tvm.const(SCALAR_MIN_FP32, dtype=dtype_x)
    elif dtype_x == "float16":
        # minimun num of float16 2**(-24)
        data_min = tvm.const(SCALAR_MIN_FP16, dtype=dtype_x)
    else:
        # minimun num of int32 1
        data_min = tvm.const(SCALAR_ONE, dtype=dtype_x)

    return _greater_equal_compare((input_x, input_y), shape_max, dtype_x, data_min)


@tbe_base.register_operator("GreaterEqual")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def greater_equal(input_x, input_y, output_z, kernel_name="greater_equal"):
    """
    do element-wise greater equal operation between two input tensors

    Parameters
    ----------
    input_x: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, support fp16,fp32,int32,uint8,int8
    input_y: dict
        dict{"shape":tuple or list,"dtype":str, range: tuple or list}
        shape of data
        the data type, support fp16,fp32,int32,uint8,int8
    output_z: dict, reserved field
        dict with keys(shape and dtype) of output
    kernel_name: str
        cce kernel name, default value is "greater_equal"

    Returns
    -------
    None
    """
    # check input tensor data_type
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "uint8", "int8")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    para_check.check_elewise_shape_range([input_x, input_y], support_broadcast=True)
    if x_dtype != y_dtype:
        error_info = {}
        error_info['errCode'] = para_check.OP_ERROR_CODE_018
        error_info['op_name'] = 'greater_equal'
        error_info['param_name1'] = 'x_dtype'
        error_info['param_name2'] = 'y_dtype'
        error_info['param1_dtype'] = str(x_dtype)
        error_info['param2_dtype'] = str(y_dtype)
        raise RuntimeError(error_info, "In op[%s], the parameter[%s][%s] are not equal in dtype with dtype[%s][%s]." % (
            error_info['op_name'], error_info['param_name1'], error_info['param_name2'], error_info['param1_dtype'],
            error_info['param2_dtype']))

    ins = tbe_base.classify([input_x, input_y], tbe_base.Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_x, input_y) in ins:
        with tbe_base.compute():
            # shape
            x_shape, y_shape = shape_util.variable_shape([input_x, input_y], support_broadcast=True)
            x_shape, y_shape = shape_util.refine_shapes_for_broadcast(x_shape, y_shape)

            # greater_equal compute
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_y = tvm.placeholder(y_shape, y_dtype, "tensor_y")
            res = greater_equal_compute(tensor_x, tensor_y, output_z, kernel_name)

            tensors.append([tensor_x, tensor_y, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
