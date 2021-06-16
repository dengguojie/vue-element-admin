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
dynamic acts_ulq_input_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


SHAPE_SIZE_LIMIT = 2 ** 31


# pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
@register_operator_compute('ActsULQInputGrad', op_mode='dynamic', support_fusion=False)
def acts_ulq_input_grad_compute(data_y_grad, data_clamp_min_mask, data_clamp_max_mask, kernel_name):
    """
    calculating grad of acts_ulq

    Parameters
    ----------
    data_y_grad: TVM tensor
        input grad
    data_clamp_min_mask: TVM tensor
        indicator where x > clamp_min
    data_clamp_max_mask: TVM tensor
        indicator where x < clamp_max
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : grade of acts_ulq
    """
    dtype = data_y_grad.dtype
    data_clamp_min_mask = tbe.cast_to(data_clamp_min_mask, dtype)
    data_clamp_max_mask = tbe.cast_to(data_clamp_max_mask, dtype)
    signal = tbe.vmul(data_clamp_min_mask, data_clamp_max_mask)
    x_grad = tbe.vmul(data_y_grad, signal)

    return x_grad


@register_operator('ActsULQInputGrad')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.KERNEL_NAME)
def acts_ulq_input_grad(y_grad, clamp_min_mask, clamp_max_mask, x_grad, kernel_name="acts_ulq_input_grad"):
    """
    calculating grad of acts_ulq

    Parameters
    ----------
    data_y_grad: TVM tensor
        input grad
    data_clamp_min_mask: TVM tensor
        indicator where x > clamp_min
    data_clamp_max_mask: TVM tensor
        indicator where x < clamp_max
    kernel_name: str
        cce kernel name, default value is acts_ulq_input_grad

    Returns
    -------
    None
    """
    y_grad_shape = y_grad.get('shape')
    y_grad_range =y_grad.get('range')
    y_grad_size = 1
    for i in range(len(y_grad_shape)):
        if y_grad_shape[i] == -1:
            if y_grad_range[i][1] is None:
                y_grad_size *= SHAPE_SIZE_LIMIT
            else:
                y_grad_size *= y_grad_range[i][1]
        else:
            y_grad_size *= y_grad_shape[i]
    if y_grad_size > SHAPE_SIZE_LIMIT:
        error_detail = "The shape size of y_grad must be smaller than {}!".format(SHAPE_SIZE_LIMIT)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, 'y_grad', error_detail)

    clamp_min_mask_shape = clamp_min_mask.get('shape')
    clamp_max_mask_shape = clamp_max_mask.get('shape')

    if clamp_min_mask_shape != y_grad_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'clamp_min_mask', 'y_grad', clamp_min_mask_shape, y_grad_shape, y_grad_shape)

    if clamp_max_mask_shape != y_grad_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'clamp_max_mask', 'y_grad', clamp_max_mask_shape, y_grad_shape, y_grad_shape)

    x_grad_shape = x_grad.get('shape')
    if x_grad_shape != y_grad_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'x_grad', 'y_grad', x_grad_shape, y_grad_shape, y_grad_shape)

    check_list = ['float16', 'float32']
    y_grad_type = y_grad.get('dtype').lower()
    clamp_min_mask_type = clamp_min_mask.get('dtype').lower()
    clamp_max_mask_type = clamp_max_mask.get('dtype').lower()
    x_grad_type = x_grad.get('dtype').lower()

    para_check.check_dtype_rule(y_grad_type, check_list, 'y_grad')
    para_check.check_dtype_rule(clamp_min_mask_type, check_list, 'clamp_min_mask')
    para_check.check_dtype_rule(clamp_max_mask_type, check_list, 'clamp_max_mask')

    if clamp_min_mask_type != clamp_max_mask_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'clamp_min_mask', 'clamp_max_mask', clamp_min_mask_type, clamp_max_mask_type)

    if y_grad_type != x_grad_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'y_grad', 'x_grad', y_grad_type, x_grad_type)

    ins = classify([y_grad, clamp_min_mask, clamp_max_mask], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (y_grad, clamp_min_mask, clamp_max_mask) in ins:
        with tbe.compute():
            shape_y_grad, shape_clamp_min_mask, shape_clamp_max_mask = shape_util.variable_shape(
                [y_grad, clamp_min_mask, clamp_max_mask])

            data_y_grad = tvm.placeholder(shape_y_grad, y_grad_type, 'data_y_grad')
            data_clamp_min_mask = tvm.placeholder(shape_clamp_min_mask, clamp_min_mask_type, 'data_clamp_min_mask')
            data_clamp_max_mask = tvm.placeholder(shape_clamp_max_mask, clamp_max_mask_type, 'data_clamp_max_mask')

            res = acts_ulq_input_grad_compute(data_y_grad, data_clamp_min_mask, data_clamp_max_mask, kernel_name)
            tensors.append([data_y_grad, data_clamp_min_mask, data_clamp_max_mask, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        'name': kernel_name,
        'print_ir': False,
        'tensor_list': tensors,
        'bool_storage_as_1bit': False}

    tbe.build(schedules, config)
