# Copyright 2021 Huawei Technologies Co., Ltd
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
relu_grad_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@register_operator_compute("ReluGradV2", op_mode="dynamic", support_fusion=True)
def relu_grad_v2_compute(gradients, mask, backprops,
                         kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).

    Parameters
    ----------
    gradients: TVM tensor
        input tensor of grad
    mask: TVM tensor
        input tensor of relu output
    backprops: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    res: TVM tensor
        the result of relu_grad_compute
    """
    dtype = gradients.dtype
    trans_type = dtype
    vsel_support = tbe_platform.api_check_support("tbe.dsl.vsel", dtype)

    # need cast int8 or uint8 to float16
    if dtype in ("int8", "uint8") or not vsel_support:
        gradients = tbe.cast_to(gradients, "float16")
        trans_type = "float16"

    result = tbe.vsel(mask, gradients, tvm.const(0, trans_type))

    # cast int8 or uint8 back
    if dtype in ("int8", "uint8") or not vsel_support:
        result = tbe.cast_to(result, dtype)

    return result


@register_operator("ReluGradV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def relu_grad_v2(gradients, mask, backprops, kernel_name="relu_grad_v2"):
    """
    calculate the backpropagation of relu operation
    output_backprops = input_gradients*1(input_features>0) or 0(input_features<=0).
    support dtype:float16,float32,int32,int8,uint8

    Parameters
    ----------
    gradients: dict
        dict of grad
    mask: dict
        dict of relu output mask
    backprops: dict
        output of relu grad
    kernel_name: str
        cce kernel name, default value is "relu_grad_v2"

    Returns
    -------
    None
    """
    shape_input_gradients = gradients.get("shape")
    shape_input_features = mask.get("shape")

    para_check.check_shape(shape_input_gradients, param_name="gradients")
    para_check.check_shape(shape_input_features, param_name="mask")

    dtype_input_gradients = gradients.get("dtype").lower()
    dtype_input_features = mask.get("dtype").lower()

    if dtype_input_features in ("uint8",):
        dtype_input_features = "uint1"

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_input_gradients, check_list, param_name="gradients")

    ins = classify([gradients, mask], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_gradients, _mask) in ins:
        with tbe.compute():
            shape_gradients, shape_mask = shape_util.variable_shape([_gradients, _mask])
            data_input_gradients = tvm.placeholder(shape_gradients,
                                                   name="data_input_gradients",
                                                   dtype=dtype_input_gradients)
            data_input_features = tvm.placeholder(shape_mask,
                                                  name="data_input_features",
                                                  dtype=dtype_input_features)
            res = relu_grad_v2_compute(data_input_gradients, data_input_features,
                                       backprops, kernel_name)

            tensors.append([data_input_gradients, data_input_features, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
