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
dynamic floor_mod
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,redefined-argument-from-local
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod"):
    """
    Compute remainder of division
    res = x1 - floor(input_data_x / input_data_y) * input_data_y

    Parameters
    ----------
    x1: TVM tensor
        input tensor has shape, dtype and range attributes
    x2: TVM tensor
        input tensor has shape, dtype and range attributes
    y: dict
        dict with keys(shape, dtype and range) of output
    kernel_name : str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    res: TVM tensor
        the calculation results
    """

    dtype = x1.dtype
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)

    shape_x, shape_y, shape = shape_util.broadcast_shapes(shape_x, shape_y,
                                                          param_name_input1="x1",
                                                          param_name_input2="x2")

    # calculate result, using float32 for better precision
    has_improve_precision = False
    input_x_fp32 = x1
    input_y_fp32 = x2
    if tbe_platform.api_check_support("te.lang.cce.vdiv",
                                      "float32"):
        input_x_fp32 = tbe.cast_to(x1, "float32")
        input_y_fp32 = tbe.cast_to(x2, "float32")
        has_improve_precision = True

    input_x_fp32 = tbe.broadcast(input_x_fp32, shape)
    input_y_fp32 = tbe.broadcast(input_y_fp32, shape)

    res = tbe.vdiv(input_x_fp32, input_y_fp32)

    if tbe_platform.api_check_support("te.lang.cce.floor",
                                      res.dtype):
        res = tbe.floor(res)
    else:
        res = tbe.cast_to(res, "float16")
        res = tbe.floor(res)

    if dtype != "int32":
        if has_improve_precision:
            res = tbe.cast_to(res, "float32")
        else:
            res = tbe.cast_to(res, "float16")
        res = tbe.vmul(res, input_y_fp32)
        res = tbe.vsub(input_x_fp32, res)
        if has_improve_precision:
            res = tbe.cast_to(res, dtype)
    else:
        x2_broad = tbe.broadcast(x2, shape)
        x1_broad = tbe.broadcast(x1, shape)
        res = tbe.vmul(res, x2_broad)
        res = tbe.vsub(x1_broad, res)

    return res


@register_operator("FloorMod")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod"):
    """
    calculate the remainder of division, support fp16,fp32,int32
    res = x1 -floor(input_data_x / input_data_y)* input_data_y

    Parameters
    ----------
    x1: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals dst_dtype, support fp16,fp32,int32
    x2: dict
        dict{"shape":tuple or list,"dtype":str, "range": tuple or list}
        shape of data
        the data type, src_dtype equals  of dst_dtype, support fp16,fp32,int32
    y: dict, reserved field
        dict with keys(shape, dtype and range) of output
    kernel_name: str
        cce kernel name, default value is "floor_mod"

    Returns
    ------
    None
    """

    # check input tensor data_type
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="x1")
    para_check.check_dtype(dtype_y, check_list, param_name="x2")

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("floor_mod", 'x1', 'x2', str(dtype_x), str(dtype_y))

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([x1, x2])
            input_data_x = tvm.placeholder(shape_x, name="input_data_x",
                                           dtype=dtype_x)
            input_data_y = tvm.placeholder(shape_y, name="input_data_y",
                                           dtype=dtype_y)
            res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name)

            tensors.append([input_data_x, input_data_y, res])
        with tvm.target.cce():
            auto_sch = tbe.auto_schedule(res)
        schedules.append(auto_sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
