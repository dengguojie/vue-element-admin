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
pt_sub
"""
import te.lang.cce as tbe
from te import tvm
from te.utils import para_check
from te.utils import shape_util
import te.platform as tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


def op_select_format(x1, x2, y, kernel_name="pt_sub"):
    dtype_list = ["float16", "float32", "int32"]
    dtype_len = len(dtype_list)

    support_format = ["ND", "NC1HWC0", "FRACTAL_Z", "FRACTAL_NZ"]
    support_dtype = []

    for dtype in dtype_list:
        support_dtype.extend([dtype] * len(support_format))

    support_format = support_format * dtype_len

    input0 = gen_param(classify="input0", name="x1",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    input1 = gen_param(classify="input1", name="x2",
                       datatype=",".join(support_dtype),
                       format=",".join(support_format))
    output0 = gen_param(classify="output0", name="y",
                        datatype=",".join(support_dtype),
                        format=",".join(support_format))

    param_list = [input0, input1, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


@tbe_platform.fusion_manager.fusion_manager.register("pt_sub")
def pt_sub_compute(x1, x2, y, broadcast_flag, kernel_name="pt_sub"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2 : TVM tensor
        the placeholder of x2
    y : dict
        dict of y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "pt_sub"

    Returns
    -------
    output tensor
    """
    # get shape of x1 and x2
    shape_x1 = tbe.util.shape_to_list(x1.shape)
    shape_x2 = tbe.util.shape_to_list(x2.shape)

    if shape_x1 != shape_x2:
        if broadcast_flag:
            # if shape not equal, then apply broadcast.
            shape_x, shape_y, shape_max = para_check.produce_shapes(shape_x1, shape_x2)
            x1 = tbe.broadcast(x1, shape_max)
            x2 = tbe.broadcast(x2, shape_max)
        else:
            x2 = tbe.broadcast(x2, shape_x1)

    res = tbe.vsub(x1, x2)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def pt_sub(x1, x2, y, kernel_name="pt_sub"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input0
    x2 : dict
        shape and dtype of input1
    y : dict
        shape and dtype of output, should be same type as input0

    kernel_name : str
        kernel name, default value is "pt_sub"

    Returns
    -------
    None
    """

    # check shape
    x1_shape = x1.get("shape")
    x2_shape = x2.get("shape")
    x1_shape = shape_util.scalar2tensor_one(x1_shape)
    x2_shape = shape_util.scalar2tensor_one(x2_shape)
    para_check.check_shape(x1_shape)
    para_check.check_shape(x2_shape)

    # check dtype
    x1_dtype = x1.get("dtype").lower()
    x2_dtype = x2.get("dtype").lower()
    dtype_list = ("float16", "float32", "int32")
    para_check.check_dtype(x1_dtype, dtype_list)
    para_check.check_dtype(x2_dtype, dtype_list)

    # check broadcast in 5HD
    x1_format = x1.get("format")
    x2_format = x2.get("format")
    format_list = ("NC1HWC0", "FRACTAL_Z", "FRACTAL_NZ")
    if x1_format in format_list and x2_format in format_list:
        if x1_shape != x2_shape and (x2_shape[0] != 1 or len(x2_shape) != 1):
            raise RuntimeError("When x1 and x2 in 5HD, do not support broadcast.")

    data_x1 = tvm.placeholder(x1_shape, dtype=x1_dtype, name="data_x1")
    data_x2 = tvm.placeholder(x2_shape, dtype=x2_dtype, name="data_x2")

    broadcast_flag = True
    # check x2 is 1D or not
    if para_check.is_scalar(x2_shape):
        broadcast_flag = False
        x2_shape = tuple([1] * (len(x1_shape) - len(x2_shape))) + tuple(x2_shape)
    data_x2 = tvm.placeholder(x2_shape, dtype=x2_dtype, name="data_x2")
    res = pt_sub_compute(data_x1, data_x2, y, broadcast_flag, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res]}
    tbe.cce_build_code(schedule, config)
