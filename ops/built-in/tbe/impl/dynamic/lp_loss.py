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
lp_loss
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from tbe.common.utils.shape_util import shape_to_list


# pylint: disable=invalid-name,unused-argument,too-many-arguments,too-many-locals
def l1_loss_compute(predict, label, dims, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param reduction: str
        reduce mode, can be 'mean','sum' or 'none'
    :return: output tensor
    """
    predict_shape = shape_to_list(predict.shape)

    # float16 cast to float32
    precision_dtype = "float32"
    predict_dtype = predict.dtype.lower()
    if predict_dtype == "float16":
        predict = tbe.cast_to(predict, precision_dtype)
        label = tbe.cast_to(label, precision_dtype)

    # calculate the result of loss = |predict-label|
    loss = tbe.vabs(tbe.vsub(predict, label))

    if reduction == "mean":
        reduce_elts = 1.0
        for i in predict_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=precision_dtype)
        else:
            cof = tbe.var("cof", dtype=precision_dtype)
            if precision_dtype == "float16":
                tbe.var("cof_empty", dtype=precision_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", precision_dtype)
        # calculate the result of sum(loss)
    if reduction == "sum":
        loss = tbe.reduce_sum(loss, dims)

    # calculate the result of mean(loss)
    elif reduction == "mean":
        div_loss = tbe.vmuls(loss, cof)
        loss = tbe.reduce_sum(div_loss, dims)
    elif reduction == "none":
        pass
    loss = tbe.cast_to(loss, predict_dtype)
    return loss


@register_operator_compute("LpLoss", op_mode="dynamic", support_fusion=True)
def lp_loss_compute(predict, label, axis, p, reduction):
    """
    :param predict: TVM tensor
        the placeholder of predict
    :param label: TVM tensor
        the placeholder of label
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: ernel name, default value is "lp_loss"
    :return: output tensor
    """
    res = l1_loss_compute(predict, label, axis, reduction)
    return res


@register_operator("LpLoss")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def lp_loss(predict, label, y, p, reduction="mean", kernel_name="lp_loss"):
    """
    :param predict: dict
        shape and dtype of input
    :param label: dict
        shape and dtype of label, should be same shape and type as predict
    :param y: dict
        shape and dtype of y, should be same shape and type as predict
    :param p: int
        decides which loss to compute, now the p only can be 1 to compute l1_loss
    :param reduction: str
        reduce mode,can be 'mean','sum' or 'none'
    :param kernel_name: kernel name, default value is "lp_loss"
    :return:
        None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    label_shape = label.get("shape")
    label_dtype = label.get("dtype").lower()

    dtype_list = ["float16", "float32"]
    reduction_list = ["none", "mean", "sum"]

    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_dtype(label_dtype, dtype_list)
    para_check.check_shape(predict_shape)
    para_check.check_shape(label_shape)

    shape_util.compare_tensor_dict_key(predict, label, "shape")
    shape_util.compare_tensor_dict_key(predict, label, "dtype")

    if p != 1:
        raise RuntimeError("lp_loss only supports l1_loss")

    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    schedules = []
    tensors = []
    axes = list(range(len(predict_shape)))
    tbe_context.get_context().add_compile_info("reduction", reduction)

    if reduction == "none":

        ins = classify([predict, label], OpPatternMode.ELEWISE)

        for (_predict, _label) in ins:
            with tbe.compute():
                shape_predict, shape_label = shape_util.variable_shape([_predict, _label])
                predict_data = tvm.placeholder(shape_predict, dtype=predict_dtype, name="predict_data")
                label_data = tvm.placeholder(shape_label, dtype=label_dtype, name="label_data")

                res = lp_loss_compute(predict_data, label_data, axes, p, reduction)
                tensors.append([predict_data, label_data, res])

            with tvm.target.cce():
                schedule = tbe.auto_schedule(res)
            schedules.append(schedule)

    else:
        predict["rel_pos_to_reduce"] = "before"
        label["rel_pos_to_reduce"] = "before"
        input_axis = {"shape":[len(axes), ], "value":axes, "rel_pos_to_reduce":"axis"}

        tbe_context.get_context().add_compile_info("reduction", reduction)

        ins = classify([predict, label, input_axis], OpPatternMode.REDUCE, {"keepdims":False})

        for (_predict, _label, _input_axis) in ins:
            with tbe.compute():
                shape_predict, shape_label = shape_util.variable_shape([_predict, _label, _input_axis],
                                                                       op_mode="reduce")[0:2]
                predict_data = tvm.placeholder(shape_predict, dtype=predict_dtype, name="predict_data")
                label_data = tvm.placeholder(shape_label, dtype=label_dtype, name="label_data")

                res = lp_loss_compute(predict_data, label_data, _input_axis["value"], p, reduction)
                tensors.append([predict_data, label_data, res])

            with tvm.target.cce():
                schedule = tbe.auto_schedule(res)
            schedules.append(schedule)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
