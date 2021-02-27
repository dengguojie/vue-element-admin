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
axpy_v2_dynamic
"""

import te.lang.cce as tbe
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base
from te.utils import shape_util
from te.utils import para_check
from te.utils.para_check import check_dtype
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute

@register_operator_compute("AxpyV2", op_mode="dynamic", support_fusion=False)
def axpy_v2_compute(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of input_x
    x2 : TVM tensor
        the placeholder of x2
    y : dict
        dict of y, include keys(shape and dtype)
    alpha : TVM tensor
        scalar of mul-factor
    kernel_name : str
        kernel name, default value is "axpy_v2"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x1 = tbe.util.shape_to_list(x1.shape)
    shape_x2 = tbe.util.shape_to_list(x2.shape)
    dtype_alpha = alpha.dtype.lower()
    dtype = x1.dtype.lower()
    precision_dtype = "float32"
    # cast dtype
    if dtype in ("float16", "float32"):
        if dtype_alpha != dtype:
            alpha = tbe.cast_to(alpha, dtype)

    if dtype == "int32":
        x1 = tbe.cast_to(x1, precision_dtype)
        x2 = tbe.cast_to(x2, precision_dtype)
        if dtype_alpha != precision_dtype:
            alpha = tbe.cast_to(alpha, precision_dtype)

    if shape_x1 != shape_x2:
        # if shape not equal, then apply broadcast.
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x1,
                                                                  shape_x2,
                                                                  param_name_input1='x1',
                                                                  param_name_input2='x2')
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)
        alpha = tbe.broadcast(alpha, shape_max)
    else:
        alpha = tbe.broadcast(alpha, shape_x1)

    res = tbe.vmla(x2, alpha, x1)
    if dtype == "int32":
        res = tbe.cast_to(res, dtype)
    return res


@register_operator("AxpyV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def axpy_v2(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data of axpy

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    alpha : dict
        shape and dtype of alpha
        scalar apply to input_y:input_y*alpha
    y : dict
        shape and dtype of output, should be same shape and type as input

    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    None
    """
    # check kernel name
    para_check.check_kernel_name(kernel_name)
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    alpha_dtype = alpha.get("dtype").lower()
    # check dtype
    dtype_list0 = ("float16", "float32", "int32")
    dtype_list1 = ("float16", "float32")
    check_dtype(dtype_x1, dtype_list0)
    check_dtype(dtype_x2, dtype_list0)
    check_dtype(alpha_dtype, dtype_list1)
    para_check.check_elewise_shape_range([x1, x2])
    ins = classify([x1, x2, alpha], Mode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for(x1, x2, alpha) in ins:
        with tbe_base.compute():
            x_shape, y_shape, alpha_shape = shape_util.variable_shape([x1, x2, alpha])
            data1 = tvm.placeholder(x_shape, dtype=dtype_x1, name="data1")
            data2 = tvm.placeholder(y_shape, dtype=dtype_x2, name="data2")
            alpha_input = tvm.placeholder(alpha_shape, dtype=alpha_dtype, name="alpha_input")
            res = axpy_v2_compute(data1, data2, alpha_input, y, kernel_name)
            tensors.append([data1, data2, alpha_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
