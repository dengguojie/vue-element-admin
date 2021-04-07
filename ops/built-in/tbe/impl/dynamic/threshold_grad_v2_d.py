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
threshold_grad_v2_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# pylint: disable=too-many-locals,unused-argument
@register_operator_compute("ThresholdGradV2D", op_mode="dynamic", support_fusion=True)
def threshold_grad_v2_d_compute(input_gradients, input_features,
                                output_backprops, threshold,
                                kernel_name="threshold_grad_v2_d"):
    """
    calculating data

    Parameters
    ----------
    input_gradients : TVM tensor
        input tensor of gradients
    input_features : TVM tensor
        input tensor of features
    output_backprops : dict
        dict of output_backprops, include keys(shape and dtype)
    threshold:
    kernel_name : str
        kernel name, default value is "threshold_grad_v2_d"

    Returns
    -------
    res: TVM tensor
        the result of threshold_grad_v2_d_compute
    """
    shape_x = shape_util.shape_to_list(input_gradients.shape)
    shape_y = shape_util.shape_to_list(input_features.shape)
    shape_x, shape_y, shape_max = \
        shape_util.broadcast_shapes(shape_x, shape_y,
                                    param_name_input1="input_x",
                                    param_name_input2="input_y")
    input_gradients = tbe.broadcast(input_gradients, shape_max)
    input_features = tbe.broadcast(input_features, shape_max)
    dtype = input_gradients.dtype
    result = tbe.vcmpsel(input_features, threshold, 'gt',
                         input_gradients, tvm.const(0, dtype))

    return result


@register_operator("ThresholdGradV2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def threshold_grad_v2_d(input_gradients, input_features, output_backprops, threshold,
                        kernel_name="threshold_grad_v2_d"):
    """
    calculating data

    Parameters
    ----------
    input_gradients : dict
        shape and dtype of input_gradients
    input_features : dict
        shape and dtype of input_features
    output_backprops : dict
        shape and dtype of output_backprops,
        should be same shape and type as inputs
    threshold : dict
        shape and dtype of threshold, 0-dimensional array
    kernel_name : str
        kernel name, default value is "threshold_grad_v2_d"

    Returns
    -------
    None
    """
    dtype_input_gradients = input_gradients.get("dtype").lower()
    dtype_input_features = input_features.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_input_gradients, check_list)
    para_check.check_dtype(dtype_input_features, check_list)
    para_check.check_elewise_shape_range([input_gradients, input_features],
                                         support_broadcast=True)
    schedules, tensors = [], []
    ins = classify([input_gradients, input_features], OpPatternMode.ELEWISE_WITH_BROADCAST)
    for (data_x, data_y) in ins:
        with tbe.compute():
            shape_input_gradients, shape_input_features = \
                shape_util.variable_shape([data_x, data_y])
            data_input_gradients = tvm.placeholder(shape_input_gradients,
                                                   name="data_input_gradients",
                                                   dtype=dtype_input_gradients)
            data_input_features = tvm.placeholder(shape_input_features,
                                                  name="data_input_features",
                                                  dtype=dtype_input_features)
            res = threshold_grad_v2_d_compute(data_input_gradients, data_input_features,
                                              output_backprops, threshold, kernel_name)
            tensors.append([data_input_gradients, data_input_features, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
