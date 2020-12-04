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
softmax_cross_entropy_with_logits
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

# limit of input dimvalue
MAX_SHAPE_NUM = 10000000


# pylint: disable=unused-argument
def get_op_support_info(input_features,
                        input_labels,
                        output_loss,
                        output_backprop,
                        kernel_name="softmax_cross_entropy_with_logits"):
    """
    get softmax_cross_entropy_with_logits slice info
    """
    shape_features = input_features.get("shape")
    shape_labels = input_labels.get("shape")

    def _get_split_info(idx):
        """
        get split_info
        """
        if shape_features[idx] == shape_labels[idx]:
            split_info = [SplitInput([0, [idx], [-1], [-1]], [1, [idx], [-1], [-1]]),
                          SplitOutput([0, [idx]], [1, [idx]])]
        elif shape_features[idx] != shape_labels[idx]:
            if shape_features[idx] == 1:
                split_info = [SplitInput([1, [idx], [-1], [-1]]),
                              SplitOutput([0, [idx]], [1, [idx]])]
            elif shape_labels[idx] == 1:
                split_info = [SplitInput([0, [idx], [-1], [-1]]),
                              SplitOutput([0, [idx]], [1, [idx]])]
            else:
                split_info = None
        return split_info

    if len(shape_features) == 4 and len(shape_labels) == 4:
        axis_split_list = []
        for idx, _ in enumerate(shape_features):
            if idx == 1:
                continue
            split_info = _get_split_info(idx)
            if split_info is not None:
                axis_split_list.append(split_info)
    elif len(shape_features) <= 2 and len(shape_labels) <= 2:
        if len(shape_features) == 1 and len(shape_labels) == 1:
            axis_split_list = None
        else:
            axis_split_list = []
            split_info = _get_split_info(0)
            if split_info is not None:
                axis_split_list.append(split_info)
    else:
        axis_split_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, None, 0, 0)

    return op_cal_info_in_json




# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("softmax_cross_entropy_with_logits")
def softmax_cross_entropy_with_logits_nchw_compute(
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

    data_max = tbe.reduce_max(input_features, axis=1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    data_sub = tbe.vsub(input_features, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(input_labels, data_log)
    data_muls = tbe.vmuls(data_mul, SCALAR_MINUS_ONE)
    loss = tbe.sum(data_muls, axis=1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    res = [loss, backprop]
    return res


@tbe_platform.fusion_manager.fusion_manager.register("softmax_cross_entropy_with_logits")
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

    # Last axis is too large, use L1 workspace compute
    # and special designed schedule
    current_csize_maximum_fp32 = 15360
    high_perf_csize_maximum_fp32 = 20000
    if current_csize_maximum_fp32 < shape_broadcast[1] < \
            high_perf_csize_maximum_fp32:
        return softmax_cross_entropy_with_logits_compute_ex(input_features,
                                                            input_labels)
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.api_check_support("te.lang.cce.vexp",
                                                "float32"):
        input_features = tbe.cast_to(input_features, "float32")
        input_labels = tbe.cast_to(input_labels, "float32")
        has_improve_precision = True

    data_max = tbe.reduce_max(input_features, axis=-1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_broadcast)
    data_sub = tbe.vsub(input_features, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=-1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_broadcast)
    data_div = tbe.vdiv(data_exp, data_sum_broadcast)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    data_log = tbe.vsub(data_sub, data_log_tmp)
    data_mul = tbe.vmul(input_labels, data_log)
    data_muls = tbe.vmuls(data_mul, SCALAR_MINUS_ONE)
    loss = tbe.sum(data_muls, axis=-1, keepdims=True)
    backprop = tbe.vsub(data_div, input_labels)

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


def softmax_cross_entropy_with_logits_compute_ex(input_features,
                                                 input_labels):
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

    if dtype == "float16":
        input_features = tbe.cast_to(input_features, "float32")
        input_labels = tbe.cast_to(input_labels, "float32")

    with tvm.tag_scope("last_axis_reduce_max"):
        reduce_axis = tvm.reduce_axis((0, shape_broadcast[1]), name="rax0")
        data_max = tvm.compute((shape_broadcast[0], 1),
                               lambda upper, lower:
                               tvm.max(input_features[upper, reduce_axis],
                                       axis=reduce_axis),
                               name="last_axis_reduce_max")
    with tvm.tag_scope("elewise_binary_sub_scalar_L1"):
        data_sub = tvm.compute(input_features.shape,
                               lambda higher, lower:
                               input_features[higher][lower] - data_max[higher][0],
                               name="manual_sub_0")
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=-1, keepdims=True)
    with tvm.tag_scope("elewise_binary_div"):
        data_div = tvm.compute(data_exp.shape,
                               lambda higher, lower:
                               data_exp[higher][lower] / data_sum[higher][0],
                               name="manual_div_0")
    data_log_tmp = tbe.vlog(data_sum)
    with tvm.tag_scope("elewise_get_L1_workspace"):
        fake_buffer = tvm.compute(data_sub.shape,
                                  lambda higher, lower: tvm.const(0, "float32"),
                                  name="get_L1_workspace")
    with tvm.tag_scope("elewise_binary_sub"):
        data_log = tvm.compute(data_sub.shape,
                               lambda higher, lower:
                               fake_buffer[higher][lower] -
                               data_log_tmp[higher][0],
                               name="manual_sub_1")
    data_mul = tbe.vmul(input_labels, data_log)
    with tvm.tag_scope("last_axis_reduce_sum_reuse"):
        reduce_axis = tvm.reduce_axis((0, shape_broadcast[1]), name="rax1")
        loss = tvm.compute((shape_broadcast[0], 1),
                           lambda upper, lower:
                           tvm.sum(data_mul[upper, reduce_axis],
                                   axis=reduce_axis),
                           name="last_axis_reduce_sum_reuse")
    loss = tbe.vmuls(loss, SCALAR_MINUS_ONE)
    backprop = tbe.vsub(data_div, input_labels)

    if dtype == "float16":
        loss = tbe.cast_to(loss, "float16")
        backprop = tbe.cast_to(backprop, "float16")

    res = [loss, backprop]

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
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
    para_check.check_shape(shape_features, param_name="input_features")
    para_check.check_shape(shape_labels, param_name="input_labels")

    check_list = ("float16", "float32")
    input_dtype = input_features.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_features")

    if len(shape_features) == 4:
        if len(shape_features) != len(shape_labels):
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                         "The length of two inputs must be same")
        if input_dtype != "float32":
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits", "Not supported dtype!")
        data_features = tvm.placeholder(shape_features, dtype=input_dtype,
                                        name="data_features")
        data_labels = tvm.placeholder(shape_labels, dtype=input_dtype,
                                      name="data_labels")
        res = softmax_cross_entropy_with_logits_nchw_compute(data_features,
                                                             data_labels,
                                                             output_loss,
                                                             output_backprop)
    else:
        if len(shape_features) == 1 and len(shape_labels) == 1:
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                         "The rank of two inputs can not be 1 at the same time")
        if len(shape_features) > 2 or len(shape_labels) > 2:
            error_manager_vector.raise_err_specific_reson("softmax_cross_entropy_with_logits",
                                                         "logits and labels must be either 2-dimensional,"
                                                         "or broadcasted to 2-dimensional")
        if len(shape_features) == 1 or len(shape_labels) == 1:
            shape_features, shape_labels, shape_broadcast = \
                shape_util.broadcast_shapes(shape_features, shape_labels, param_name_input1="input_features",
                                    param_name_input2="input_labels")

        data_features = tvm.placeholder(shape_features, dtype=input_dtype,
                                        name="data_features")
        data_labels = tvm.placeholder(shape_labels, dtype=input_dtype,
                                      name="data_labels")
        res = softmax_cross_entropy_with_logits_compute(data_features,
                                                        data_labels,
                                                        output_loss,
                                                        output_backprop)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    tensor_list = [data_features, data_labels] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
