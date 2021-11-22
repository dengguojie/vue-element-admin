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
dynamic acts_ulq
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SHAPE_SIZE_LIMIT = 2 ** 31
    CALCULATE_TYPE = 'float32'


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
# 'pylint: disable=invalid-name,too-many-locals,too-many-statements
@register_operator_compute('ActsULQ', op_mode='dynamic', support_fusion=False)
def acts_ulq_compute(data, clamp_min, clamp_max, fixed_min, step, kernel_name):
    """
    calculating data's add, c = a + b

    Parameters
    ----------
    data. TVM tensor
        the placeholder of input data
    clamp_min: TVM tensor
        the placeholder of clamp min
    clamp_max: TVM tensor
        the placeholder of clamp max
    fixed_min: bool
        attr, indicate whether fix clamp min to zero
    kernel_name: str
        cce kernel name, default value is add

    Returns
    -------
    res : output tensors
    """
    shape_data = shape_util.shape_to_list(data.shape)
    shape_clamp_min = shape_util.shape_to_list(clamp_min.shape)
    shape_clamp_max = shape_util.shape_to_list(clamp_max.shape)
    eps = 1.192092896e-07

    _, _, shape_broadcast = shape_util.broadcast_shapes(
        shape_data, shape_clamp_min, param_name_input1='data', param_name_input2='clamp_min')
    _, _, shape_broadcast = shape_util.broadcast_shapes(
        shape_data, shape_clamp_max, param_name_input1='data', param_name_input2='clamp_max')

    dtype = data.dtype
    data = tbe.cast_to(data, Constant.CALCULATE_TYPE)
    clamp_min = tbe.cast_to(clamp_min, Constant.CALCULATE_TYPE)
    clamp_max = tbe.cast_to(clamp_max, Constant.CALCULATE_TYPE)

    if fixed_min:
        ori_clip_min = tbe.vmuls(clamp_min, tvm.const(0, Constant.CALCULATE_TYPE))
    else:
        ori_clip_min = tbe.vmins(clamp_min, tvm.const(0, Constant.CALCULATE_TYPE))

    ori_clip_max = tbe.vmaxs(clamp_max, tvm.const(step * eps, Constant.CALCULATE_TYPE))

    scale = tbe.vsub(ori_clip_max, ori_clip_min)
    scale = tbe.vdiv(scale, tbe.broadcast(tvm.const(step, Constant.CALCULATE_TYPE), scale.shape))

    offset = tbe.vdiv(ori_clip_min, scale)
    offset = tbe.round(offset)
    offset = tbe.cast_to(offset, 'float16')
    offset = tbe.cast_to(offset, Constant.CALCULATE_TYPE)

    #fake quant clip min/max
    clip_min = tbe.vmul(scale, offset)
    clip_max = tbe.vadds(offset, tvm.const(step, Constant.CALCULATE_TYPE))
    clip_max = tbe.vmul(clip_max, scale)

    clip_min = tbe.broadcast(clip_min, shape_broadcast)
    clip_max = tbe.broadcast(clip_max, shape_broadcast)
    clamped_x = tbe.vmax(data, clip_min)
    clamped_x = tbe.vmin(clamped_x, clip_max)

    clamp_min_mask = tbe.vcmp(data, clip_min, 'ge')
    clamp_max_mask = tbe.vcmp(data, clip_max, 'le')

    #fake quant x
    raw_x = tbe.vdiv(clamped_x, tbe.broadcast(scale, shape_broadcast))
    round_x = tbe.round(raw_x)
    round_x = tbe.cast_to(round_x, 'float16')
    round_x = tbe.cast_to(round_x, Constant.CALCULATE_TYPE)
    clamped_loss = tbe.vsub(round_x, raw_x)
    clamped_loss = tbe.vdiv(clamped_loss, tbe.broadcast(tvm.const(step, Constant.CALCULATE_TYPE), shape_broadcast))

    raw_m = tbe.vdiv(ori_clip_min, scale)
    round_m = tbe.round(raw_m)
    round_m = tbe.cast_to(round_m, 'float16')
    round_m = tbe.cast_to(round_m, Constant.CALCULATE_TYPE)
    loss_m = tbe.vsub(round_m, raw_m)
    loss_m = tbe.vdiv(loss_m, tbe.broadcast(tvm.const(step, Constant.CALCULATE_TYPE), shape_broadcast))

    clamped_loss_float16 = tbe.cast_to(clamped_loss, 'float16')
    loss_m_float16 = tbe.broadcast(loss_m, shape_broadcast)
    loss_m_float16 = tbe.cast_to(loss_m_float16, 'float16')
    clamped_loss = tbe.vsel(clamp_min_mask, clamped_loss_float16, loss_m_float16)
    clamped_loss = tbe.vsel(clamp_max_mask, clamped_loss_float16, loss_m_float16)
    clamped_loss = tbe.cast_to(clamped_loss, dtype)

    output = tbe.vmul(round_x, tbe.broadcast(scale, shape_broadcast))
    output = tbe.cast_to(output, dtype)

    clamp_min_mask = tbe.vsel(clamp_min_mask, tvm.const(1, 'float16'), tvm.const(0, 'float16'))
    clamp_max_mask = tbe.vsel(clamp_max_mask, tvm.const(1, 'float16'), tvm.const(0, 'float16'))
    clamp_min_mask = tbe.cast_to(clamp_min_mask, dtype)
    clamp_max_mask = tbe.cast_to(clamp_max_mask, dtype)

    return [output, clamp_min_mask, clamp_max_mask, clamped_loss]


# 'pylint: disable=too-many-branches,too-many-statements
@register_operator('ActsULQ')
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
    para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def acts_ulq(
    data, clamp_min, clamp_max, output, clamp_min_mask, clamp_max_mask, x_clamped_loss, fixed_min=False, num_bits=8,
    kernel_name='acts_ulq'):
    """
    algorithm: ulq

    Parameters
    ----------
    data: dict
        shape and dtype of feature map, only support float16, float32
    clamp_min: dict
        shape and dtype of clamp min, only support float16, float32
    clamp_max: dict
        shape and dtype of clamp max, only support float16, float32
    y: dict
        shape and dtype of output
    clamp_min_mask:
        mask if data > clamp_min (fake quant)
    clamp_max_mask:
        mask if data < clamp_max (fake quant)
    x_clamped_loss:
        loss
    kernel_name : str
        cce kernel name, default value is acts_ulq

    Returns
    -------
    None
    """
    if num_bits != 8:
        error_manager_vector.raise_err_input_param_not_in_range(kernel_name, 'num_bits', 8, 8, num_bits)

    data_shape = data.get('shape')
    data_range = data.get('range')
    data_size = 1
    for index, data_shape_value in enumerate(data_shape):
        if data_shape_value == -1:
            if data_range[index][1] is None:
                data_size *= Constant.SHAPE_SIZE_LIMIT
            else:
                data_size *= data_range[index][1]
        else:
            data_size *= data_shape_value
    if data_size > Constant.SHAPE_SIZE_LIMIT:
        error_detail = "The shape size of data must be smaller than {}!".format(Constant.SHAPE_SIZE_LIMIT)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, 'data', error_detail)

    clamp_min_shape = shape_util.scalar2tensor_one(clamp_min.get('shape'))
    clamp_max_shape = shape_util.scalar2tensor_one(clamp_max.get('shape'))

    if clamp_min_shape != clamp_max_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'clamp_min', 'clamp_max', clamp_min_shape, clamp_max_shape, clamp_min_shape)

    for i in clamp_min_shape:
        if i != 1:
            error_detail = "The clamp_min and the clamp_max should be all one!"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, 'clamp_min', 'clamp_max', error_detail)

    output_shape = output.get('shape')
    clamp_min_mask_shape = clamp_min_mask.get('shape')
    clamp_max_mask_shape = clamp_max_mask.get('shape')
    x_clamped_loss_shape = x_clamped_loss.get('shape')

    if output_shape != data_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'y', 'data', output_shape, data_shape, data_shape)
    if clamp_min_mask_shape != data_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'clamp_min_mask', 'data', clamp_min_mask_shape, data_shape, data_shape)
    if clamp_max_mask_shape != data_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'clamp_max_mask', 'data', clamp_max_mask_shape, data_shape, data_shape)
    if x_clamped_loss_shape != data_shape:
        error_manager_vector.raise_err_inputs_shape_not_equal(
            kernel_name, 'x_clamped_loss', 'data', x_clamped_loss_shape, data_shape, data_shape)

    check_list = ['float16', 'float32']
    data_type = data.get('dtype').lower()
    clamp_min_type = clamp_min.get('dtype').lower()
    clamp_max_type = clamp_max.get('dtype').lower()
    output_type = output.get('dtype').lower()
    clamp_min_mask_type = clamp_min_mask.get('dtype').lower()
    clamp_max_mask_type = clamp_max_mask.get('dtype').lower()
    x_clamped_loss_type = x_clamped_loss.get('dtype').lower()

    para_check.check_dtype_rule(data_type, check_list, 'data')

    if clamp_min_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'clamp_min', 'data', clamp_min_type, data_type)

    if clamp_max_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'clamp_max', 'data', clamp_max_type, data_type)

    if output_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'y', 'data', output_type, data_type)
    if clamp_min_mask_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'clamp_min_mask', 'data', clamp_min_mask_type, data_type)
    if clamp_max_mask_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'clamp_max_mask', 'data', clamp_max_mask_type, data_type)
    if x_clamped_loss_type != data_type:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            kernel_name, 'x_clamped_loss', 'data', x_clamped_loss_type, data_type)

    n = 2 ** num_bits - 1
    ins = classify([data, clamp_min, clamp_max], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _clamp_min, _clamp_max) in ins:
        with tbe.compute():
            shape_x, shape_clamp_min, shape_clamp_max = shape_util.variable_shape([_x, _clamp_min, _clamp_max])

            data_x = tvm.placeholder(shape_x, data_type, 'data_x')
            data_clamp_min = tvm.placeholder(shape_clamp_min, data_type, 'data_clamp_min')
            data_clamp_max = tvm.placeholder(shape_clamp_max, data_type, 'data_clamp_max')

            res = acts_ulq_compute(data_x, data_clamp_min, data_clamp_max, fixed_min, n, kernel_name)

            tensor_list = [data_x, data_clamp_min, data_clamp_max] + list(res)
            tensors.append(tensor_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        'name': kernel_name,
        'print_ir': False,
        'tensor_list': tensors,
        'bool_storage_as_1bit': False}

    tbe.build(schedules, config)
