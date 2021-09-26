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
dynamic expand
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode

 
# pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Expand", op_mode="dynamic", support_fusion=False)
def expand_compute(x, shape):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    x: list of placeholders.
        Input data.
    shape : list or tuple.
        Number of the axis replicates.
    shape: dict.
        dict of output.
    Returns
    -------
    res
    """
    shape_in = x.shape
    _, _, shape_max = shape_util.broadcast_shapes(shape_in, shape)

    dtype = x.dtype

    output_tensor = tbe.broadcast(x, shape_max, dtype)
    
    # func: to avoid null compute process
    if shape_max == shape_in:
        output_tensor = tbe.vadds(output_tensor, 0)

    return output_tensor



# pylint: disable=too-many-locals,too-many-statements
@register_operator("Expand")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def expand(x, shape, y, kernel_name="expand"):
    """algorithm: expand.
    The expand in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The expand op in TBE is compatible with the tensorflow operator BroadcastTo
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    shape : dict
        shape and dtype of multiples
    y: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "expand".

    Returns
    -------
    None
    """

    input_x_dtype = x.get("dtype").lower()
    input_shape_dtype = shape.get("dtype").lower()

    input_x_shape = list(x.get("shape"))
    input_shape_shape = list(shape.get("shape"))

    check_list = ('float16', 'float32', 'int8', 'uint8', 'int32')
    para_check.check_dtype(input_x_dtype, check_list, param_name = "x")
    check_list = ('int32', 'int64')
    para_check.check_dtype(input_shape_dtype, check_list, param_name = "shape")

    if len(input_shape_shape) > 1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape should be 1D")

    input_x_range = list(x.get("range"))
    dims_value = input_shape_shape[0]

    if dims_value < -1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape[0] should be more than -1")

    if dims_value == -1:
        shape["shape"] = [-1] * len(input_x_shape)
        shape["range"] = [(1, None)] * len(input_x_shape)
    else:
        shape["shape"] = [-1] * dims_value
        shape["range"] = [(1, None)] * dims_value
    
    if len(x['range']) == 0:
        # x's range should not be empty when x is static.
        x['range'] = [(val, val) for val in x['shape']]
    else:
        x_range = []
        # x's range should not include zero.
        for range_val in x['range']:
            if range_val[0] == 0:
                x_range.append((1, range_val[1]))
            else:
                x_range.append(range_val)
        x['range'] = x_range

    extra_params = {"disable_optimization":True}
    ins = classify([shape, x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_shape, _x) in ins:
        with tbe.compute():
            shape_shape, shape_x = shape_util.variable_shape([_shape, _x])
            shape_input = tvm.placeholder(shape_shape, name="shape_input", dtype=input_shape_dtype)
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
            res = expand_compute(x_input, shape_shape)
            tensors.append([x_input, shape_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
