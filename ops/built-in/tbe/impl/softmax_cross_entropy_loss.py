# Copyright 2019 Huawei Technologies Co., Ltd
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
softmax_cross_entropy_loss
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te import tvm
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

# compute needed,scalar -1
SCALAR_MINUS_ONE = -1


# 'pylint: disable=unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("softmax_cross_entropy_loss")
def softmax_cross_entropy_loss_compute(
        scores,
        labels,
        weights,
        loss,
        log_prop,
        reduction,
        weights_flag,
        kernel_name="softmax_cross_entropy_loss"):
    """Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    scores: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    labels: TVM tensor
        input tensor contains shape and dtype attributes.
        labels data type support "int32", "int64".
    weights: TVM tensor
        If given, it has to be a 1D Tensor
    loss: dict
        when reduction=none:TVM tensor, output tensor
        when reduction=sum/mean, A Scalar
        Must have the same type as 'scores'.
    log_prop: dict
        data of output.
        Must have the same type as 'scores'.
    reduction: str
        reduce configuration mean/sum/none. Default: mean
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_loss"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "scores".
    """
    shape_scores = shape_util.shape_to_list(scores.shape)
    shape_labels = shape_util.shape_to_list(labels.shape)

    dtype = scores.dtype.lower()

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        scores = tbe.cast_to(scores, "float32")
        has_improve_precision = True

    data_max = tbe.reduce_max(scores, axis=1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_scores)
    data_sub = tbe.vsub(scores, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_scores)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    log_prop = tbe.vsub(data_sub, data_log_tmp)

    input_labels = tbe.broadcast(labels, shape_scores)
    data_mul = tbe.vmul(input_labels, log_prop)
    data_muls = tbe.vmuls(data_mul, SCALAR_MINUS_ONE)
    if weights_flag:
        input_weights = tbe.broadcast(weights, shape_scores)
        data_loss = tbe.vmul(data_muls, input_weights)
        loss = tbe.sum(data_loss, axis=1, keepdims=False)
    else:
        loss = tbe.sum(data_muls, axis=1, keepdims=False)

    loss = loss_compute(loss, reduction, shape_labels)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        log_prop = tbe.cast_to(log_prop, "float16")

    res = [loss, log_prop]

    return res


def loss_compute(loss, reduction, shape_labels):
    reduce_elts = 1.0
    for i in shape_labels:
        reduce_elts *= i
    cof = reduce_elts**(-1)

    # get total axis for reduce
    axis_d = []
    for i in range(len(shape_labels) - 1):
        axis_d.append(i)

    if reduction == 'mean':
        # calcu mean
        loss = tbe.sum(loss, axis=axis_d, keepdims=False)
        loss = tbe.vmuls(loss, cof)
    elif reduction == 'sum':
        # calcu sum
        loss = tbe.sum(loss, axis=axis_d, keepdims=False)
    elif reduction == 'none':
        loss = loss

    return loss


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def softmax_cross_entropy_loss(
        scores,
        labels,
        weights,
        loss,
        log_prop,
        ignore_index = 0,
        reduction='mean',
        kernel_name="softmax_cross_entropy_loss"):
    """
    Computes softmax cross entropy cost.

    Parameters
    ----------
    scores: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    labels: dict
        input tensor contains shape and dtype attributes.
        labels data type support "int32", "int64".
    weights: dict
        A manual rescaling weight given to each class.
        If given, it has to be a 1D Tensor assigning weight to each of the classes.
        Otherwise, it is treated as if having all ones.
    loss: dict
        data of output.
        Must have the same type as 'scores'.
    log_prop: dict
        data of output.
        Must have the same type as 'scores'.
    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
        It's an optional value.
    reduction: str (default is mean)
        Type of reduction to apply to loss: none, sum, mean(default). 
        'none': no reduction will be applied,
        'sum': the output will be summed.
        'mean': the sum of the output will be divided by the number of elements in the output.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_loss"

    Returns:
    None
    """
    shape_scores = scores.get("shape")
    shape_labels = shape_util.shape_to_list(labels.get("shape"))
    shape_labels.insert(1, 1)
    shape_labels = tuple(shape_labels)
    input_dtype = scores.get("dtype").lower()
    labels_dtype = labels.get("dtype").lower()
    if weights:
        shape_weights = shape_util.shape_to_list(weights.get("shape"))
        list_new = [1 for i in range(len(shape_scores) - 1)]
        list_new.insert(1, shape_weights[0])
        shape_weights = tuple(list_new)
        para_check.check_shape(shape_weights, param_name="weights")
        data_weights = tvm.placeholder(shape_weights, dtype=input_dtype,
                                name="data_weights")
        weights_flag = True
    else:
        weights = None
        weights_flag = False

    para_check.check_shape(shape_scores, param_name="scores")
    para_check.check_shape(shape_labels, param_name="labels")

    check_list = ("float16", "float32","float64","bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="scores")

    data_scores = tvm.placeholder(shape_scores, dtype=input_dtype,
                                    name="data_scores")
    data_labels = tvm.placeholder(shape_labels, dtype=labels_dtype,
                                    name="data_labels")
    res = softmax_cross_entropy_loss_compute(data_scores,
                                                    data_labels,
                                                    data_weights,
                                                    loss,
                                                    log_prop,
                                                    reduction,
                                                    weights_flag)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [data_scores, data_labels, data_weights] + list(res)

    config = {"name": kernel_name,
                "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)