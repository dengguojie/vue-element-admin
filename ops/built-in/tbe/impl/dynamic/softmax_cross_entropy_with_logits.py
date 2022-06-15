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


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # compute needed,scalar -1
    SCALAR_MINUS_ONE = -1


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("SoftmaxCrossEntropyWithLogits", op_mode="dynamic", support_fusion=False)
def softmax_cross_entropy_with_logits_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        mode,
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
        input_features_broad = tbe.broadcast(input_features, shape_broadcast, dtype)
        input_features = input_features_broad
        input_labels_broad = tbe.broadcast(input_labels, shape_broadcast, dtype)
        input_labels = input_labels_broad
    else:
        shape_broadcast = shape_features


    data_max = tbe.reduce_max(input_features, axis=-1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_features_cast_fp32 = tbe.cast_to(input_features, "float32")
        input_features = input_features_cast_fp32
        input_labels_cast_fp32 = tbe.cast_to(input_labels, "float32")
        input_labels = input_labels_cast_fp32
        has_improve_precision = True

    if has_improve_precision:
        data_max_broadcast = tbe.cast_to(data_max_broadcast, "float32")

    data_sub = tbe.vsub(input_features, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(data_log, input_labels)
    data_muls = tbe.vmuls(data_mul, Constant.SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("SoftmaxCrossEntropyWithLogits", op_mode="dynamic", support_fusion=False)
def softmax_cross_entropy_with_logits_workspace_compute(
        input_features,
        input_labels,
        output_loss,
        output_backprop,
        mode,
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
    input_features1 = input_features
    input_labels1 = input_labels

    if list(shape_features) != list(shape_labels):
        shape_features, shape_labels, shape_broadcast = \
            shape_util.broadcast_shapes(shape_features, shape_labels, param_name_input1="input_features",
                                        param_name_input2="input_labels")
        input_features_broad = tbe.broadcast(input_features, shape_broadcast, dtype)
        input_features_broad1 = tbe.broadcast(input_features1, shape_broadcast, dtype)
        input_features = input_features_broad
        input_features1 = input_features_broad1
        input_labels_broad = tbe.broadcast(input_labels, shape_broadcast, dtype)
        input_labels_broad1 = tbe.broadcast(input_labels1, shape_broadcast, dtype)
        input_labels = input_labels_broad
        input_labels1 = input_labels_broad1
    else:
        shape_broadcast = shape_features

    data_max = tbe.reduce_max(input_features, axis=-1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        input_labels_cast_fp32 = tbe.cast_to(input_labels, "float32")
        input_labels = input_labels_cast_fp32

        input_features1_cast_fp32 = tbe.cast_to(input_features1, "float32")
        input_features1 = input_features1_cast_fp32
        input_labels1_cast_fp32 = tbe.cast_to(input_labels1, "float32")
        input_labels1 = input_labels1_cast_fp32

        has_improve_precision = True

    if has_improve_precision:
        data_max_broadcast = tbe.cast_to(data_max_broadcast, "float32")

    data_sub = tbe.vsub(input_features1, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.reduce_sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(data_log, input_labels)
    data_muls = tbe.vmuls(data_mul, Constant.SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels1)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]
    res += [data_sub, data_exp, data_sum_broadcast]
    return res


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator("SoftmaxCrossEntropyWithLogits", pattern="SoftmaxCrossEntropyWithLogits")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
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

    if len(shape_features) == 1 and len(shape_labels) == 2:
        shape_features = [1, shape_features[0]]
        input_features['range'] = [[1, 1], input_features['range'][0]]
        input_features['shape'] = shape_features
    if len(shape_features) == 2 and len(shape_labels) == 1:
        shape_labels = [1, shape_labels[0]]
        input_labels['range'] = [[1, 1], input_labels['range'][0]]
        input_labels['shape'] = shape_labels

    shape_util.compare_tensor_dict_key(input_features, input_labels, "dtype")

    check_list = ("float16", "float32")
    input_dtype = input_features.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_features")

    shape_features = shape_util.scalar2tensor_one(shape_features)
    shape_labels = shape_util.scalar2tensor_one(shape_labels)

    input_features["shape"] = shape_features
    input_labels["shape"] = shape_labels
    # 0: not need align, 1: low dim_axis align, 2: high dim_axis align
    extra_params = {
        "dimension_align_ward": [2, 2]
    }
    ins = classify([input_features, input_labels], "SoftmaxNorm", extra_params)

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
            shape_features, shape_labels = shape_util.variable_shape([x1, x2], op_mode="softmax_norm")
            data_features = tvm.placeholder(shape_features, dtype=input_dtype, name="data_features")
            data_labels = tvm.placeholder(shape_labels, dtype=input_dtype, name="data_labels")
            if "cut" not in x1.get("mode"):
                res = softmax_cross_entropy_with_logits_compute(data_features, data_labels, output_loss,
                                                                output_backprop, x1["mode"])
            else:
                res = softmax_cross_entropy_with_logits_workspace_compute(data_features, data_labels, output_loss,
                                                                          output_backprop, x1["mode"])
            tensor_list = [data_features, data_labels] + list(res)
            if len(tensor_list) < 7:
                dummpy_placeholder_num = 7 - len(tensor_list)
                for i in range(dummpy_placeholder_num):
                    dummpy_placeholder = tvm.placeholder(output_backprop.get("shape"), dtype=input_dtype,
                                                         name="dummy_placeholder" + str(i))
                    tensor_list.append(dummpy_placeholder)
            tensors.append(tensor_list)
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res[:2])
        schedules.append(schedule)

    tbe_context.get_context().add_compile_info("common_info",
                                               {"ub_size": tbe_platform.get_soc_spec("UB_SIZE"),
                                                "core_num": tbe_platform.get_soc_spec("CORE_NUM")})
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "dummy_placeholder": True}
    tbe.build(schedules, config)
