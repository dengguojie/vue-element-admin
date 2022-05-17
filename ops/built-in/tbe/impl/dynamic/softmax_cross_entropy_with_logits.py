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
from impl.util.platform_adapter import operation


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # compute needed,scalar -1
    SCALAR_MINUS_ONE = -1
    MAX_INT32_VALUE = 2147483647


def _process_range(range0, range1):
    dim00_range = range0[0]
    dim01_range = range0[1]
    dim10_range = range1[0]
    dim11_range = range1[1]
    if _range_to_int(dim00_range[0]) > 1 and _range_to_int(dim10_range[0]) > 1:
        intersection_dim00_dim10_range = (max(_range_to_int(dim00_range[0]), _range_to_int(dim10_range[0])),
                                          min(_range_to_int(dim00_range[1]), _range_to_int(dim10_range[1])))
        dim00_range = intersection_dim00_dim10_range
        dim10_range = intersection_dim00_dim10_range
    else:
        dim00_range = (_range_to_int(dim00_range[0]), _range_to_int(dim00_range[1]))
        dim10_range = (_range_to_int(dim10_range[0]), _range_to_int(dim10_range[1]))

    if _range_to_int(dim01_range[0]) > 1 and _range_to_int(dim11_range[0]) > 1:
        intersection_dim01_dim11_range = (max(_range_to_int(dim01_range[0]), _range_to_int(dim11_range[0])),
                                          min(_range_to_int(dim01_range[1]), _range_to_int(dim11_range[1])))
        dim01_range = intersection_dim01_dim11_range
        dim11_range = intersection_dim01_dim11_range
    else:
        dim01_range = (_range_to_int(dim01_range[0]), _range_to_int(dim01_range[1]))
        dim11_range = (_range_to_int(dim11_range[0]), _range_to_int(dim11_range[1]))

    range0 = [dim00_range, dim01_range]
    range1 = [dim10_range, dim11_range]
    return range0, range1


def _range_to_int(range_val):
    return Constant.MAX_INT32_VALUE if range_val is None else int(range_val)


# 'pylint: disable=too-many-locals,too-many-statements
def variable_shape(inputs: list, support_broadcast=False):
    """
    :param inputs: all inputs
    :param support_broadcast: whether to support broadcast
    :return:
    """
    def _fill(_inputs):
        x_0, x_1 = _inputs
        shape0, range0 = list(x_0["shape"]), list(x_0["range"])
        shape1, range1 = list(x_1["shape"]), list(x_1["range"])

        range0, range1 = _process_range(range0, range1)
        swapped = False
        if len(shape0) < len(shape1):
            shape0, range0, shape1, range1 = shape1, range1, shape0, range0
            swapped = True
        d_v = len(shape0) - len(shape1)
        shape1 = [1] * d_v + shape1
        range1 = [(1, 1)] * d_v + range1
        if swapped:
            shape0, range0, shape1, range1 = shape1, range1, shape0, range0
        return [shape0, shape1], [range0, range1]

    def _maybe_broadcast():
        for _r in ranges:
            if _r[i][0] <= 1:
                return True
        return False

    mode = inputs[0].get("mode")
    if mode is None:
        mode = para_check.ORIGINAL
    operation.get_context().add("mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("_mode", mode)
        ori_axis = inputs[0].get("ori_axis")
        if ori_axis is not None:
            current_compute.add("ori_axis", ori_axis)
        axis_dtype = inputs[0].get("axis_dtype")
        if axis_dtype is not None:
            current_compute.add("axis_dtype", axis_dtype)
    operation.get_context().add("support_broadcast", support_broadcast)

    shapes, ranges = _fill(inputs)

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast() and "copy" not in mode
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1 and _range[i][0] == _range[i][1]:
                operation.var("dim_" + str(_suffix) + "_" + str(i), (1, Constant.MAX_INT32_VALUE))
                d_shape.append(_range[i][0])
            elif shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var("dim_" + str(_suffix) + "_" + str(i), _range[i])
                else:
                    operation.var("dim_" + str(_suffix) + "_" + str(i), _range[i])
                d_shape.append(_var)
            else:
                operation.var("dim_" + str(_suffix) + "_" + str(i), (1, Constant.MAX_INT32_VALUE))
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes


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

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        input_features_cast_fp32 = tbe.cast_to(input_features, "float32")
        input_features = input_features_cast_fp32
        input_labels_cast_fp32 = tbe.cast_to(input_labels, "float32")
        input_labels = input_labels_cast_fp32
        has_improve_precision = True

    fp32_use_fp16_reduce_max = False
    if input_features.dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.reduce_max", "float32"):
        input_features_cast_fp16 = tbe.cast_to(input_features, "float16")
        input_features = input_features_cast_fp16
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
    data_muls = tbe.vmuls(data_mul, Constant.SCALAR_MINUS_ONE)
    loss = tbe.reduce_sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    is_data_features_broadcast = mode in ("vec6_and_cut", "vec9_and_cut", "vec1_and_cut", "vec2_and_cut")
    is_data_labels_broadcast = mode in ("vec6_and_cut", "vec9_and_cut", "vec4_and_cut", "vec8_and_cut")
    is_workspace = "cut" in mode
    res = [loss, backprop]
    if is_workspace:
        res += [data_sub, data_exp, data_sum_broadcast]
    if is_data_features_broadcast and dtype == "float32":
        res.append(input_features_broad)
    if is_data_labels_broadcast and dtype == "float32":
        res.append(input_labels_broad)
    if dtype == "float16":
        res.append(input_features_cast_fp32)
        res.append(input_labels_cast_fp32)
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

    input_features['range'], input_labels['range'] = _process_range(input_features['range'],
                                                                    input_labels['range'])

    shape_util.compare_tensor_dict_key(input_features, input_labels, "dtype")

    check_list = ("float16", "float32")
    input_dtype = input_features.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_features")

    shape_features = shape_util.scalar2tensor_one(shape_features)
    shape_labels = shape_util.scalar2tensor_one(shape_labels)

    input_features["shape"] = shape_features
    input_labels["shape"] = shape_labels

    operation.get_context().add("ub_size", tbe_platform.get_soc_spec("UB_SIZE"))

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
            shape_features, shape_labels = variable_shape([x1, x2], support_broadcast=True)
            data_features = tvm.placeholder(shape_features, dtype=input_dtype, name="data_features")
            data_labels = tvm.placeholder(shape_labels, dtype=input_dtype, name="data_labels")
            res = softmax_cross_entropy_with_logits_compute(data_features, data_labels, output_loss, output_backprop,
                                                            x1["mode"])
            tensor_list = [data_features, data_labels] + list(res)
            if len(tensor_list) < 9:
                dummpy_placeholder_num = 9 - len(tensor_list)
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
