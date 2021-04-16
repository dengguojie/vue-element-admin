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
dynamic softmax_cross_entropy_with_logits
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode

# compute needed,scalar -1
SCALAR_MINUS_ONE = -1

# limit of input dimvalue
MAX_SHAPE_NUM = 10000000


@register_operator_compute("SoftmaxCrossEntropyWithLogits", op_mode="dynamic", support_fusion=False)
def softmax_cross_entropy_with_logits_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits"):
    """
    Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    input_features: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: TVM tensor
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "input_features".
    """
    shape_features = shape_util.shape_to_list(input_features.shape)
    shape_labels = shape_util.shape_to_list(input_labels.shape)
    dtype = input_features.dtype.lower()

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(shape_features, shape_labels, param_name_input1="input_features",
                                        param_name_input2="input_labels")
        input_features = tbe.broadcast(input_features, shape_broadcast,
                                       dtype)
        input_labels = tbe.broadcast(input_labels, shape_broadcast,
                                     dtype)
    else:
        shape_broadcast = shape_features

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        input_features = tbe.cast_to(input_features, "float32")
        input_labels = tbe.cast_to(input_labels, "float32")
        has_improve_precision = True

    fp32_use_fp16_reduce_max = False
    if input_features.dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.reduce_max", "float32"):
        input_features = tbe.cast_to(input_features, "float16")
        fp32_use_fp16_reduce_max = True

    data_max = tbe.reduce_max(input_features, axis=-1, keepdims=True)

    if fp32_use_fp16_reduce_max:
        data_max = tbe.cast_to(data_max, "float32")

    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    data_sub = tbe.vsub(input_features, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(input_labels, data_log)
    data_muls = tbe.vmuls(data_mul, SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


@register_operator("SoftmaxCrossEntropyWithLogits", pattern="SoftmaxCrossEntropyWithLogits")
def softmax_cross_entropy_with_logits(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        kernel_name="softmax_cross_entropy_with_logits"):
    """
    Computes softmax cross entropy cost.

    Parameters
    ----------
    input_features: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    input_labels: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'input_features'.
    output_loss: dict
        data of output.
        Must have the same type as 'input_features'.
    output_backprop: dict
        data of output.
        Must have the same type as 'input_features'.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_with_logits"

    Returns:
    None
    """
    shape_features = input_features.get("shape")
    shape_labels = input_labels.get("shape")

    shape_util.compare_tensor_dict_key(input_features, input_labels, "dtype")

    check_list = ("float16", "float32")
    input_dtype = input_features.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_features")
    para_check.check_elewise_shape_range([input_features, input_labels], support_broadcast=True)

    shape_features = shape_util.scalar2tensor_one(shape_features)
    shape_labels = shape_util.scalar2tensor_one(shape_labels)

    input_features["shape"] = shape_features
    input_labels["shape"] = shape_labels

    ins = classify([input_features, input_labels], "softmax_cross_entropy_with_logits_with_reduce")

    if len(shape_features) == 1 and len(shape_labels) == 1:
        error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                      "The rank of two inputs can not be 1 at the same time")
    if len(shape_features) > 2 or len(shape_labels) > 2:
        error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                      "logits and labels must be either 2-dimensional,"
                                                      "or broadcasted to 2-dimensional")

    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            shape_features, shape_labels = shape_util.variable_shape([x1, x2])
            shape_features, shape_labels = shape_util.refine_shapes_for_broadcast(shape_features, shape_labels)
            data_features = tvm.placeholder(shape_features, dtype=input_dtype, name="data_features")
            data_labels = tvm.placeholder(shape_labels, dtype=input_dtype, name="data_labels")
            res = softmax_cross_entropy_with_logits_compute(data_features, data_labels, output_loss, output_backprop)
            tensor_list = [data_features, data_labels] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)
    tbe_context.get_context().add_compile_info("ori_shape",
                                               {"features_shape0": input_features['shape'][0],
                                                "features_shape1": input_features['shape'][1],
                                                "labels_shape0": input_labels['shape'][0],
                                                "labels_shape1": input_labels['shape'][1]})
    tbe_context.get_context().add_compile_info("common_info",
                                               {"ub_size": tbe_platform.get_soc_spec("UB_SIZE"),
                                                "core_num": tbe_platform.get_soc_spec("CORE_NUM")})
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
