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
threshold_v2_d
"""

from te.lang.cce.te_compute.elewise_compute import vcmpsel as _vcmpsel
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("ThresholdV2D", op_mode="dynamic", support_fusion=False)
# pylint: disable=unused-argument,too-many-locals,invalid-name
def threshold_v2_d_compute(x, y, threshold, value,
                           kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """
    dtype_x = x.dtype
    has_improve_precision = False
    if dtype_x in ("int8", "int32", "uint8"):
        x = tbe.cast_to(x, "float32")
        has_improve_precision = True

    threshold = tvm.const(threshold, "float32")
    value = tvm.const(value, "float32")

    data_res = _vcmpsel(x, threshold, operation='gt', slhs=x, srhs=value)
    if has_improve_precision:
        data_res = tbe.cast_to(data_res, dtype_x)
    return data_res

# pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator("ThresholdV2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def threshold_v2_d(x,
                   y,
                   threshold, value,
                   kernel_name="threshold_v2_d_cce"):
    """
    Thresholds each element of the input Tensor
    y = (x > threshold) ? x : value

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    threshold : float
        scale value to threshold at
    value : float
        scale value to replace with
    kernel_name : str
        kernel name, default value is "threshold_v2_d_cce"

    Returns
    -------
    output tensor
    """

    # get the shape and dtype
    dtype_x = x.get("dtype").lower()

    # check whether dtypes are right
    check_list = ("float16", "float32", 'int8', 'uint8', 'int32')
    para_check.check_dtype(dtype_x, check_list)

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])[0]
            input_data = tvm.placeholder(x_shape, name="input_data",
                                         dtype=dtype_x)
            res = threshold_v2_d_compute(input_data, y, threshold, value, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
