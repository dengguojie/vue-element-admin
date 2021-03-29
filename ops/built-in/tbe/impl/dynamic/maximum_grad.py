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
maximum_grad
"""
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


def _compare_value_int32(data_x, data_y, shape_dz):
    """
    The input data type of this function only support int32;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    min_value_int = tvm.const(1, dtype="int32")
    data_zero_int = tvm.const(0, dtype="int32")
    min_value_tensor = tbe.broadcast(min_value_int, shape_dz)
    data_zero_int_tensor = tbe.broadcast(data_zero_int, shape_dz)
    sub_xy = tbe.vsub(data_x, data_y)
    add_min = tbe.vadd(sub_xy, min_value_tensor)
    vmax_zero = tbe.vmax(add_min, data_zero_int_tensor)
    result = tbe.vmin(vmax_zero, min_value_tensor)

    return result


# pylint: disable = locally-disabled,invalid-name,too-many-arguments
# pylint: disable = unused-argument
def _compare_value_float(data_x, data_y):
    """
    The input data type of the function only support float;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    # The smallest positive subnormal number of float32 is 2**(-126)
    min_value = tvm.const(2 ** (-126), dtype="float32")
    # (2**(-126))*(2**(62))*(2**(62))*(2**(2)) = 1
    # so min_value*max_value*max_value*max_value_1 = 1
    max_value = tvm.const(2 ** (62), dtype="float32")
    max_value_1 = tvm.const(2 ** (2), dtype="float32")

    data_zero = tbe.vmuls(data_x, 0)
    min_value_tensor = tbe.vadds(data_zero, min_value)
    max_value_tensor = tbe.vadds(data_zero, max_value)
    max_value_1_tensor = tbe.vadds(data_zero, max_value_1)
    sub_xy = tbe.vsub(data_x, data_y)
    add_min_value = tbe.vadds(sub_xy, min_value)
    vmax_zero = tbe.vmax(add_min_value, data_zero)
    vmin_min_value = tbe.vmin(vmax_zero, min_value_tensor)
    vmul_max_value = tbe.vmul(vmin_min_value, max_value_tensor)
    vmul_max_value_1 = tbe.vmul(vmul_max_value, max_value_tensor)
    result = tbe.vmul(vmul_max_value_1, max_value_1_tensor)

    return result


def _compare_value(data_x, data_y, dtype, shape_dz):
    """
    The input data type of the function only support float and int32;
    The return value of the function: if data_x >= data_y return 1;
    else return 0.
    """
    if dtype == "int32":
        compare_value_data = _compare_value_int32(data_x, data_y, shape_dz)
    else:
        compare_value_data = _compare_value_float(data_x, data_y)

    return compare_value_data


def _calculate_result_ge(data_x, data_y, data_dz, dtype, shape_dz):
    """
    The input data type of the function only support float int32 dtype;
    The return value of the function: if data_y >= data_x :
    result_dx = data_dz, result_dy = 0;
    else result_dx = 0,result_dx = data_dz.
    """
    minus_one = tvm.const(-1, dtype="float32")
    value_one = tvm.const(1, dtype="float32")
    if dtype == "int32":
        minus_one = tvm.const(-1, dtype="int32")
        value_one = tvm.const(1, dtype="int32")
    minus_one_tensor = tbe.broadcast(minus_one, shape_dz)
    value_one_tensor = tbe.broadcast(value_one, shape_dz)
    # if data_y < data_x ; datax_select_ge = 1; else datax_select_le =0;
    datax_select_ge = _compare_value(data_x, data_y, dtype, shape_dz)
    result_dx = tbe.vmul(data_dz, datax_select_ge)
    select_reverse = tbe.vsub(datax_select_ge, value_one_tensor)
    select_dy = tbe.vmul(select_reverse, minus_one_tensor)
    result_dy = tbe.vmul(data_dz, select_dy)

    return result_dx, result_dy


def _reduce_result(shape_x, shape_y, shape_dz, result_dx, result_dy):
    """
    If the shapes of the two input data are not equal,
    we need to call this function to do reduce operation.
    """
    if shape_x != shape_dz:
        reduce_axis = []
        for i, shape_x_i in enumerate(shape_x):
            if shape_x_i == 1:
                reduce_axis.append(i)
        result_dx = tbe.reduce_sum(result_dx, axis=reduce_axis, keepdims=None)

    if shape_y != shape_dz:
        reduce_axis = []
        for i, shape_y_i in enumerate(shape_y):
            if shape_y_i == 1:
                reduce_axis.append(i)
        result_dy = tbe.reduce_sum(result_dy, axis=reduce_axis, keepdims=None)

    return result_dx, result_dy


@register_operator_compute("MaximumGrad", op_mode="dynamic", support_fusion=True)
def maximum_grad_compute(data_x, data_y, data_dz, y1, y2, grad_x, grad_y,
                         kernel_name="maximum_grad"):
    """
    algorithm:
    calculating maximum_grad of the two input data

    Parameters
    ----------
    data_x:TVM tensor.
        the placeholder of data_x
    data_y:TVM tensor.
        the placeholder of data_y
    data_dz:TVM tensor.
        the placeholder of data_dz
    y1: dict:
        dict with keys(shape and dtype) of y1
    y2: dict:
        dict with keys(shape and dtype) of y2
    kernel_name: str
        cce kernel name, default value is "maximum_grad"

    Returns:
    -------
    results of maximum or maximum_grad of the two input data.
    """
    dtype = data_x.dtype
    if data_x.dtype == "float16":
        data_x = tbe.cast_to(data_x, "float32")
        data_y = tbe.cast_to(data_y, "float32")
        data_dz = tbe.cast_to(data_dz, "float32")

    shape_dz = shape_util.shape_to_list(data_dz.shape)
    shape_x = shape_util.shape_to_list(data_x.shape)
    shape_y = shape_util.shape_to_list(data_y.shape)
    data_x = tbe.broadcast(data_x, shape_dz)
    data_y = tbe.broadcast(data_y, shape_dz)

    result_dx, result_dy = _calculate_result_ge(data_x, data_y, data_dz,
                                                dtype, shape_dz)

    if shape_x != shape_dz or shape_y != shape_dz:
        if dtype == "int32":
            rule_desc = "sum not support int32"
            error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, \
                                                              "data_x", dtype)
        result_dx, result_dy = _reduce_result(shape_x, shape_y, shape_dz,
                                              result_dx, result_dy)

    if dtype == "float16":
        result_dx = tbe.cast_to(result_dx, "float16")
        result_dy = tbe.cast_to(result_dy, "float16")

    res = [result_dx, result_dy]

    return res


@register_operator("MaximumGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def maximum_grad(grads, x1, x2, y1, y2, grad_x=True, grad_y=True,
                 kernel_name="maximum_grad"):
    """
    algorithm: maximum_grad
    calculating the reversed outputs of the function "maximum"
    "maximum" : z = vmin(x,y),  dx, dy = maximum_grad(...)

    Parameters
    ----------
    x1: dict
        dict with keys(shape and dtype) of x1
    x2: dict
        dict with keys(shape and dtype) of x2
    grads: dict
        dict with keys(shape and dtype) of grads
    y1: dict:
        dict with keys(shape and dtype) of y1
    y2: dict:
        dict with keys(shape and dtype) of y2
    kernel_name: str
        kernel name, default value is "maximum_grad"

    Returns:
    -------
    none.
    """

    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    dtype_dz = grads.get("dtype").lower()

    if dtype_x != dtype_y != dtype_dz:
        rule_desc = "the dtypes of intputs should be same"
        param_value = "%s,%s,%s" % (dtype_x, dtype_y, dtype_dz)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, \
                                                          "grads,x1,x2", param_value)

    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype_dz, check_list, param_name="grads")
    para_check.check_dtype(dtype_x, check_list, param_name="x1")
    para_check.check_dtype(dtype_y, check_list, param_name="x2")
    ins = classify([grads, x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (grads, x1, x2) in ins:
        with tbe.compute():
            shape_dz, shape_x, shape_y = shape_util.variable_shape([grads, x1, x2])
            data_dz = tvm.placeholder(shape_dz, dtype=dtype_dz, name="data_dz")
            data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")
            res = maximum_grad_compute(data_x, data_y, data_dz, y1, y2, grad_x,
                                       grad_y, kernel_name)
            tensors.append([data_dz, data_x, data_y] + res)
        with tvm.target.cce():
            if (grad_x, grad_y) == (True, False):
                sch = tbe.auto_schedule(res[0])
                res = [res[0]]
            if (grad_x, grad_y) == (False, True):
                sch = tbe.auto_schedule(res[1])
                res = [res[1]]
            if (grad_x, grad_y) == (True, True):
                sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
