# Copyright 2021 Huawei Technologies Co., Ltd
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
binary_cross_entropy
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from te.lang.base.shape_classifier import classify
from te.lang.base.shape_classifier import Mode
import te.lang.base as tbe_base
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute



# eps value
SCALAR_EPS = 1e-12
# the type of None
NoneType = type(None)


# pylint: disable=invalid-name,too-many-arguments
# pylint: disable=unused-argument,too-many-locals
@register_operator_compute("BinaryCrossEntropy", op_mode="dynamic", support_fusion=False)
def binary_cross_entropy_compute(x, y, weight, output,
                                 reduction, kernel_name):
    """
    calculating binary_cross_entropy

    Parameters
    ----------
    x : TVM tensor
        the output of previous layer
    y : TVM tensor
        label
    weight :
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output :
        loss result after compute
    reduction :
        reduce configuration parameter: mean/sum/none. Default: mean
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    result : TVM tensor
        output tensor
    """
    ori_dtype = x.dtype
    trans_dtype = ori_dtype
    shape = shape_util.shape_to_list(x.shape)
    if ori_dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
        if weight is not None:
            weight = tbe.cast_to(weight, "float32")
        trans_dtype = "float32"

    const_one = tvm.const(1, trans_dtype)
    const_neg_one = tvm.const(-1, trans_dtype)
    # calcu value : y * log(x)
    x = tbe.vmaxs(x, tvm.const(SCALAR_EPS, trans_dtype))
    x_log_tmp = tbe.vlog(x, priority_flag=1)
    data_mul1 = tbe.vmul(x_log_tmp, y)
    # calcu value : (1-y) * log(1-x)
    x_neg_tmp = tbe.vmuls(x, const_neg_one)
    x1_tmp = tbe.vadds(x_neg_tmp, const_one)
    y_neg_tmp = tbe.vmuls(y, const_neg_one)
    y1_tmp = tbe.vadds(y_neg_tmp, const_one)
    x1_tmp = tbe.vmaxs(x1_tmp, tvm.const(SCALAR_EPS, trans_dtype))
    x1_log_tmp = tbe.vlog(x1_tmp, priority_flag=1)
    data_mul2 = tbe.vmul(x1_log_tmp, y1_tmp)
    # calcu value : y * log(x) + (1-y) * log(1-x)
    data_sum = tbe.vadd(data_mul1, data_mul2)
    # calcu value : -(y * log(x) + (1-y) * log(1-x))
    result = tbe.vmuls(data_sum, const_neg_one)

    if weight is not None:
        result = tbe.vmul(result, weight)
    calc_dtype = trans_dtype
    if reduction == "mean":
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=calc_dtype)
        else:
            cof = tbe_base.var("cof", dtype=calc_dtype)
            if calc_dtype == "float16":
                tbe_base.var("cof_empty", dtype=calc_dtype)
            tbe_base.add_compile_info("reduce_mean_cof_dtype", calc_dtype)

    # get total axis for reduce
    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = shape_util.axis_check(len(shape), axis_d)

    if reduction == "mean":
        result = tbe.vmuls(result, cof)
        result = tbe.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "sum":
        result = tbe.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "none":
        pass

    if ori_dtype == "float16":
        result = tbe.cast_to(result, "float16")

    return result


@register_operator("BinaryCrossEntropy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def binary_cross_entropy(x, y, weight, output,
                         reduction="mean",
                         kernel_name="binary_cross_entropy"):
    """
    calculating data
    res = -w (y ln(x) + (1-y) ln(1-x))
    if reduction == sum:  res = reduce_sum(res)            output a scalar
    if reduction == mean:  res = reduce_sum(res)/data_len  output a scalar
    if reduction == none: res = res   output a tensor

    Parameters
    ----------
    x : dict
        shape and dtype of tensor predict
    y : dict
        shape and dtype of tensor target,
        should be same shape and dtype as predict
    weight : None or TVM tensor
        a manual rescaling weight given to the loss of each batch element.
        If given, has to be a Tensor of size nbatch
    output : dict
        shape and dtype of output, loss result after compute
    reduction : str
        Specifies the reduction to apply to the output:'none' | 'mean' | 'sum'
         Default: 'mean'
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number
                of elements in the output
        'sum': the output will be summed. Note: size_average and reduce
               are in the process of being deprecated
               and in the meantime, specifying either of those
               two args will override reduction.
    kernel_name : str
        kernel name, default value is "binary_cross_entropy"

    Returns
    -------
    None
    """
    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc,
                                                          "reduction", reduction)

    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    predict_dtype_lower = predict_dtype.lower()

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    target_dtype_lower = target_dtype.lower()

    # check dtype
    dtype_list = ("float16", "float32")
    para_check.check_dtype(predict_dtype, dtype_list, param_name="x")
    para_check.check_dtype(target_dtype, dtype_list, param_name="y")
    shape_util.compare_tensor_dict_key(x, y, "dtype")

    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        weight_dtype_lower = weight_dtype.lower()
        para_check.check_dtype(weight_dtype, dtype_list, param_name="weight")
        shape_util.compare_tensor_dict_key(x, weight, "dtype")

    schedules, tensors = [], []
    if weight is not None:
        ins = classify([x, y, weight], Mode.REDUCE, {"keepdims": True})
        for (_predict_shape, _target_shape, _weight_shape) in ins:
            with tbe_base.compute():
                predict_shape, target_shape, weight_shape = shape_util.variable_shape([_predict_shape,
                                                                                       _target_shape,
                                                                                       _weight_shape])
                data_weight = tvm.placeholder(weight_shape, name="data_weight",
                                              dtype=weight_dtype_lower)
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype_lower)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype_lower)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output,
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, data_weight, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([x, y], Mode.REDUCE, {"keepdims": True})
        for (_predict_shape, _target_shape) in ins:
            with tbe_base.compute():
                predict_shape, target_shape = shape_util.variable_shape([_predict_shape, _target_shape])
                data_weight = None
                data_predict = tvm.placeholder(predict_shape, name="data_predict",
                                               dtype=predict_dtype_lower)
                data_target = tvm.placeholder(target_shape, name="data_target",
                                              dtype=target_dtype_lower)
                res = binary_cross_entropy_compute(data_predict, data_target,
                                                   data_weight, output,
                                                   reduction, kernel_name)
                tensors.append([data_predict, data_target, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)
