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
dynamic logsoftmax_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify


# pylint: disable = locally-disabled,unused-argument
@register_operator("LogSoftmaxV2")
def log_softmax_v2_compute(input_x, output_y, axis=-1, kernel_name="log_softmax_v2"):
    """
    process of calculating data's log_softmax, x - log(sum(exp(x)))
    this x is x - xmax

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

    Returns
    -------
    result: TVM tensor.
    """
    inp_dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)

    data_max = tbe.reduce_max(input_x, axis=axis, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape)
    data_sub = tbe.vsub(input_x, data_max_broadcast)

    # increase accuracy
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        data_sub = tbe.cast_to(data_sub, "float32")
        has_improve_precision = True

    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=axis, keepdims=True)
    data_log = tbe.vlog(data_sum)
    data_log_broadcast = tbe.broadcast(data_log, shape)
    res = tbe.vsub(data_sub, data_log_broadcast)

    # cast output type same as input type
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("LogSoftmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def log_softmax_v2(input_x, output_y, axis=-1, kernel_name="log_softmax_v2"):
    """
    algorithm: log_softmax
    calculating data's log_softmax, x - log(sum(exp(x)))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis: int, list or tuple
        the data's axis, range is [-d, d-1]
    kernel_name : str
        cce kernel name, default value is log_softmax_v2

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
            output = log_softmax_v2_compute(input_x, output_y, axis.get("value"), kernel_name)
            tensors.append([input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
