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
sigmoid_cross_entropy_with_logits
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
import te.lang.base as tbe_base
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
from te.utils.op_utils import variable_shape
from te.utils.op_utils import refine_shapes_for_broadcast
from te.utils.op_utils import KERNEL_NAME
from te.utils.op_utils import REQUIRED_INPUT
from te.utils.op_utils import REQUIRED_OUTPUT
from te.utils.op_utils import check_op_params
from te.utils.op_utils import broadcast_shapes
from te.utils.op_utils import check_elewise_shape_range
from topi import generic

# define a scalar, value = 1
SCALAR_ONE = 1
# define a scalar, value = 0
SCALAR_ZREO = 0


# pylint: disable=locally-disabled,unused-argument,too-many-locals
# pylint: disable=invalid-name
def get_broadcast_shapes(input1, input2, input1_name, input2_name):
    """
    get_broadcast_shapes
    """
    x0_shape = shape_util.shape_to_list(input1.shape)
    x1_shape = shape_util.shape_to_list(input2.shape)
    x0_shape, x1_shape, y_shape = broadcast_shapes(x0_shape, x1_shape, param_name_input1=input1_name,
                                                   param_name_input2=input2_name)
    return y_shape


# pylint: disable=locally-disabled,unused-argument,too-many-locals
def sigmoid_cross_entropy_with_logits_compute(predict, target, loss, kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data
          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

        For x < 0, to avoid overflow in exp(-x), we reformulate the above
          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

        max(x, 0) - x * z + log(1 + exp(-abs(x)))
    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    loss : dict
        dict of loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    output tensor
    """
    y_shape = get_broadcast_shapes(predict, target, 'predict', 'target')
    predict = tbe.broadcast(predict, y_shape)
    target = tbe.broadcast(target, y_shape)
    predict_dtype = predict.dtype
    target_dtype = target.dtype
    if predict_dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vsub", "float32"):
        predict = tbe.cast_to(predict, "float32")
    if target_dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vmul", "float32"):
        target = tbe.cast_to(target, "float32")

    dtype_predict = predict.dtype
    shape_predict = shape_util.shape_to_list(predict.shape)

    const_zero = tvm.const(SCALAR_ZREO, dtype=dtype_predict)
    max_predict_zero = tbe.vmaxs(predict, const_zero)

    abs_predict = tbe.vabs(predict)
    const_zero_broadcast = tbe.broadcast(const_zero, shape_predict)
    reverse_abs_predict = tbe.vsub(const_zero_broadcast, abs_predict)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        reverse_abs_predict = tbe.cast_to(reverse_abs_predict, "float16")

    vexp_predict = tbe.vexp(reverse_abs_predict)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        vexp_predict = tbe.cast_to(vexp_predict, "float32")

    const_one = tvm.const(SCALAR_ONE, dtype=dtype_predict)
    vadds_res = tbe.vadds(vexp_predict, const_one)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        vadds_res = tbe.cast_to(vadds_res, "float16")

    vlog_res = tbe.vlog(vadds_res, priority_flag=1)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        vlog_res = tbe.cast_to(vlog_res, "float32")

    vmul_res = tbe.vmul(predict, target)
    res = tbe.vsub(vlog_res, vmul_res)
    loss = tbe.vadd(res, max_predict_zero)

    if predict_dtype == "float16":
        loss = tbe.cast_to(loss, "float16")

    return loss


@tbe_base.register_operator("SigmoidCrossEntropyWithLogits")
@check_op_params(REQUIRED_INPUT, REQUIRED_INPUT, REQUIRED_OUTPUT, KERNEL_NAME)
def sigmoid_cross_entropy_with_logits(predict, target, loss, kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data:
        calculating sigmoid cross entropy given logits
        `predict` and `target` must have the same type and shape.

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    loss : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    None
    """

    dtype_predict = predict.get("dtype")
    input_dtype_predict = dtype_predict.lower()

    dtype_target = target.get("dtype")
    input_dtype_target = dtype_target.lower()

    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype_predict, check_list, param_name="predict")
    para_check.check_dtype(input_dtype_target, check_list, param_name="target")
    check_elewise_shape_range([predict, target], support_broadcast=False)
    ins = classify([predict, target], Mode.ELEWISE)

    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe_base.compute():
            x_shape, y_shape = variable_shape([x1, x2], support_broadcast=False)
            shape_predict, shape_target = refine_shapes_for_broadcast(x_shape, y_shape)
            data_predict = tvm.placeholder(shape_predict,
                                           name="data_predict",
                                           dtype=input_dtype_predict)
            data_target = tvm.placeholder(shape_target,
                                          name="data_target",
                                          dtype=input_dtype_target)
            loss = sigmoid_cross_entropy_with_logits_compute(data_predict,
                                                             data_target,
                                                             loss,
                                                             kernel_name)

            tensors.append([data_predict, data_target, loss])
        with tvm.target.cce():
            sch = generic.auto_schedule(loss)
        schedules.append(sch)
    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
