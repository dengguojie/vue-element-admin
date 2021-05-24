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
dynamic softmax_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import classify


@register_operator("SoftmaxV2")
def softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="softmax_v2"):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
       the data's axis, range == [-d, d-1]
    kernel_name: str
        cce kernel name, default value is softmax_v2

    Returns
    -------
    output: TVM tensor
        the result of softmax
    """

    shape = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype
    axis = list(axis)
    last_dim = len(input_x.shape) - 1
    vcmax_flag = False

    for i in axis:
        if (i == -1) or (i == last_dim):
            vcmax_flag = True
    has_improve_precision = False
    if dtype == "float16":
        has_improve_precision = True
        input_x = tbe.cast_to(input_x, "float32")
        data_max = tbe.reduce_max(input_x, axis=axis, keepdims=True)
    else:
        data_max = tbe.reduce_max(input_x, axis=axis, keepdims=True)

    data_max = tbe.broadcast(data_max, shape)
    data_subtrac = tbe.vsub(input_x, data_max)
    data_exp = tbe.vexp(data_subtrac)
    data_expsum = tbe.reduce_sum(data_exp, axis, keepdims=True)
    data_expsum = tbe.broadcast(data_expsum, shape)
    output = tbe.vdiv(data_exp, data_expsum)
    if has_improve_precision and dtype == "float16":
        output = tbe.cast_to(output, "float16")

    return output


@register_operator("SoftmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT), para_check.KERNEL_NAME)
def softmax_v2(input_x, output_y, axis=-1, kernel_name="softmax_v2"):
    """
    algorithm: softmax
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x : dict
        format: FORMAT_ND , NC1HWC0
               dtype: only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
        the data's axis.
        format: FORMAT_ND, NC1HWC0
                range == [-d, d-1]
    kernel_name : str
        cce kernel name, default value is softmax_v2
    impl_mode: str.
        high_precision or high_performance for inference, default value is OpImplMode.HIGH_PERFORMANCE.
        no need to add into ops_info file.

    Returns
    -------
    None
    """

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    if not isinstance(axis, int):
        axis = list(axis)

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")
    axis = shape_util.axis_check(len(shape), axis)
    if isinstance(axis, int):
        axis = [axis]
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([input_x, input_axis], "norm")

    for (x, axis) in ins:
        with tbe.compute():
            shape_var_new, _= shape_util.variable_shape([x, axis], op_mode="norm")
            input_x = tvm.placeholder(shape_var_new, dtype=dtype, name="input_x")
            output = softmax_v2_compute(input_x, output_y, axis.get("value"), kernel_name)
            tensors.append([input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
