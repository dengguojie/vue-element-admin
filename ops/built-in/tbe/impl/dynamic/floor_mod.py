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


class Constant:
    """
    Define constant in this class
    """
    FP32_MAX_VALID = 2 ** 24


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=too-many-locals,redefined-argument-from-local
def floor_mod_compute(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
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
    def _mod(x1, x2):
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

        res_quot = tbe.vdiv(input_x_fp32, input_y_fp32)

        if tbe_platform.api_check_support("te.lang.cce.floor",
                                          res_quot.dtype):
            res_quot = tbe.floor(res_quot)
        else:
            res_quot = tbe.cast_to(res_quot, "float16")
            res_quot = tbe.floor(res_quot)

        if dtype != "int32":
            if has_improve_precision:
                result = tbe.cast_to(res_quot, "float32")
            else:
                result = tbe.cast_to(res_quot, "float16")
            result = tbe.vmul(result, input_y_fp32)
            res_rem = tbe.vsub(input_x_fp32, result)
            if has_improve_precision:
                res_rem = tbe.cast_to(res_rem, dtype)
        else:
            x2_broad = tbe.broadcast(x2, shape)
            x1_broad = tbe.broadcast(x1, shape)
            result = tbe.vmul(res_quot, x2_broad)
            res_rem = tbe.vsub(x1_broad, result)

        return res_quot, res_rem

    if impl_mode == "high_performance":
        _, res = _mod(x1, x2)
    else:
        # x1 can not be converted to a fp32 number absolute equality when its dtype is int32 and value is bigeer
        # than 2^24 sometimes, so we use 2^24 as an intermediate constant to get the exact result.
        fp32_max_valid_tensor = tbe.broadcast(Constant.FP32_MAX_VALID, shape)
        quot_x_tmp, res_x_tmp = _mod(x1, fp32_max_valid_tensor)
        quot_tmp_y, res_tmp_y = _mod(fp32_max_valid_tensor, x2)
        res = tbe.vmul(quot_x_tmp, res_tmp_y)
        res = tbe.vadd(res, res_x_tmp)
        _, res = _mod(res, x2)

    return res


@register_operator("FloorMod")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def floor_mod(x1, x2, y, kernel_name="floor_mod", impl_mode="high_performance"):
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
    impl_mode: str
        impl_mode, default value is "high_performance"

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
            res = floor_mod_compute(input_data_x, input_data_y, y, kernel_name, impl_mode)

            tensors.append([input_data_x, input_data_y, res])
        with tvm.target.cce():
            auto_sch = tbe.auto_schedule(res)
        schedules.append(auto_sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
