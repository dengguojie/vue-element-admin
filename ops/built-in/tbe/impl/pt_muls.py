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
pt_muls
"""
import functools
import te.lang.cce as tbe
from te import tvm
import te.platform as tbe_platform
from te.utils import shape_util
from te.utils import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


def op_select_format(x1, x2, y, kernel_name="pt_muls"):
    """
    Dynamic matching format

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    y : dict
        shape and dtype of output, should be same shape and type as input

    kernel_name : str
        kernel name, default value is "pt_muls"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("ori_shape")
    shape_x2 = x2.get("ori_shape")

    shape_x1 = shape_util.scalar2tensor_one(shape_x1)
    shape_x2 = shape_util.scalar2tensor_one(shape_x2)

    enum_x1 = functools.reduce(lambda x, y: x * y, shape_x1)
    enum_x2 = functools.reduce(lambda x, y: x * y, shape_x2)

    dtype_list = ("float16", "float32", "int32")

    if enum_x1 == 1 or enum_x2 == 1:
        format_list = ("ND", "NCHW", "NHWC", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0")
        dtype_list_total = functools.reduce(lambda x, y: x + y, [[ele] * len(format_list) for ele in dtype_list])
        format_list_for_non_one = format_list * len(dtype_list)
        if enum_x1 == 1:
            format_list_for_one = [x1.get("format")] * len(format_list) * len(dtype_list)
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_list_total),
                               format=",".join(format_list_for_one))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_list_total),
                               format=",".join(format_list_for_non_one))
        else:
            format_list_for_one = [x2.get("format")] * len(format_list) * len(dtype_list)
            input0 = gen_param(classify="input0", name="x1",
                               datatype=",".join(dtype_list_total),
                               format=",".join(format_list_for_non_one))
            input1 = gen_param(classify="input1", name="x2",
                               datatype=",".join(dtype_list_total),
                               format=",".join(format_list_for_one))

        output0 = gen_param(classify="output0", name="y",
                               datatype=",".join(dtype_list_total),
                               format=",".join(format_list_for_non_one))

        param_list = [input0, input1, output0]
        param_dynamic_in_json = get_dynamic_param_in_json(param_list)

        return param_dynamic_in_json

    raise RuntimeError("The shape of input0 or input1 must be 1.")


@tbe_platform.fusion_manager.fusion_manager.register("pt_muls")
def pt_muls_compute_dim(x1, x2, y, kernel_name="pt_muls"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of input_x
    x2 : TVM tensor
        the placeholder of x2, 0D or 1D tensor
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "pt_muls"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x1 = shape_util.shape_to_list(x1.shape)
    x2 = tbe.broadcast(x2, shape_x1)
    res = tbe.vmul(x1, x2)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def pt_muls(x1, x2, y, kernel_name="pt_muls"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    y : dict
        shape and dtype of output, should be same shape and type as input

    kernel_name : str
        kernel name, default value is "pt_muls"

    Returns
    -------
    None
    """

    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
    x1_dtype = x1.get("dtype").lower()
    x2_dtype = x2.get("dtype").lower()
    x1_format = x1.get("format")
    x2_format = x2.get("format")

    # check kernel name
    para_check.check_kernel_name(kernel_name)

    # check dtype
    dtype_list = ("float16", "float32", "int32")
    para_check.check_dtype(x1_dtype, dtype_list)
    para_check.check_dtype(x2_dtype, dtype_list)

    # check shape
    x1_shape = shape_util.scalar2tensor_one(x1_shape)
    x2_shape = shape_util.scalar2tensor_one(x2_shape)
    para_check.check_shape(x1_shape)
    para_check.check_shape(x2_shape)
    # check broadcast in 5HD
    format_list = ("NC1HWC0", "FRACTAL_Z", "FRACTAL_NZ")
    if x1_format in format_list and x2_format in format_list:
        if x1_shape != x2_shape and (x2_shape[0] != 1 or len(x2_shape) != 1):
            raise RuntimeError("When x1 and x2 in 5HD, do not support broadcast.")

    # check x2 is 0D or 1D tensor
    if not para_check.is_scalar(x2_shape):
        raise RuntimeError("Input x2 must be 0D or 1D.")

    data_x1 = tvm.placeholder(x1_shape, dtype=x1_dtype, name="data_x1")
    x2_shape = tuple([1] * (len(x1_shape) - len(x2_shape))) + tuple(x2_shape)
    data_x2 = tvm.placeholder(x2_shape, dtype=x2_dtype, name="data_x2")
    res = pt_muls_compute_dim(data_x1, data_x2, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    tbe.cce_build_code(schedule, config)
