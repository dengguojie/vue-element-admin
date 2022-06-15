# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic bn_training_update
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import OpAttr


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,invalid-name,unused-argument
def get_op_support_info(x,
                        sum,
                        square_sum,
                        scale,
                        offset,
                        mean,
                        variance,
                        y,
                        mean_out,
                        variance_out,
                        batch_mean,
                        batch_variance,
                        factor,
                        epsilon,
                        kernel_name="bn_training_update"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance):
    """
    Function to check if the dtype is in line with norms.

    Parameters
    ----------
    dtype_x: str
        x's data type
    dtype_sum: str
        sum's data type
    dtype_square_sum: str
        square_sum's data type
    dtype_scale: str
        scale's data type
    dtype_offset: str
        offset's data type
    dtype_mean: str
        mean's data type
    dtype_variance: str
        variance's data type

    Returns
    -------
    None
    """
    para_check.check_dtype(dtype_x, ("float16", "float32"), param_name="x")
    para_check.check_dtype(dtype_sum, ("float32",), param_name="sum")
    para_check.check_dtype(dtype_square_sum, ("float32",), param_name="square_sum")
    para_check.check_dtype(dtype_scale, ("float32",), param_name="scale")
    para_check.check_dtype(dtype_offset, ("float32",), param_name="offset")
    para_check.check_dtype(dtype_mean, ("float32",), param_name="mean")
    para_check.check_dtype(dtype_variance, ("float32",), param_name="variance")


# 'pylint: disable=too-many-branches,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator_compute("BNTrainingUpdate", op_mode="dynamic", support_fusion=True)
def bn_training_update_compute(x,
                               sum,
                               square_sum,
                               scale,
                               offset,
                               mean,
                               variance,
                               y,
                               mean_out,
                               variance_out,
                               batch_mean,
                               batch_variance,
                               factor,
                               epsilon,
                               kernel_name="bn_training_update",
                               reduce_shape=None,
                               dyn_flag=True):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum: TVM tensor
        contains square_sum data
    scale: TVM tensor
        contains scale data
    offset: TVM tensor
        contains offset data
    mean: TVM tensor
        contains mean data
    variance: TVM tensor
        contains variance data
    y: dict
        dict of output, A 'Tensor'. Has the same type as 'x'.
    mean_out: dict
        dict of mean, A 'Tensor'. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A 'Tensor'.
        The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A 'Tensor'.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A 'Tensor'.
        Has the same type as 'batch_mean'.
    factor: float
        A ratio to caculate the update mean or variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update"
    reduce_shape: list
        reduce shape of input shape
    dyn_flag: bool
        flag of dynamic or static shape

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_compute
    """
    if isinstance(factor, float):
        factor_reverse = 1.0 - factor
    else:
        factor_reverse = tbe.var("factor_reverse", dtype="float32")
        tbe_context.get_context().add_compile_info("has_factor_reverse", True)

    factor = get_attr_by_cls(factor, OpAttr(0, "factor", "Float", 0.2), "float32")
    epsilon = get_attr_by_cls(epsilon, OpAttr(1, "epsilon", "Float", 0.0000001), "float32")

    shape_x = shape_util.shape_to_list(x.shape)

    if not dyn_flag:
        data_format = y.get("format").upper()
        if not reduce_shape and data_format in ("NC1HWC0",):
            reduce_dims = [shape_x[0], shape_x[2], shape_x[3]]
        elif not reduce_shape and data_format in ("NDC1HWC0",):
            reduce_dims = [shape_x[0], shape_x[1], shape_x[3], shape_x[4]]
        else:
            reduce_dims = reduce_shape

        num = 1
        for dim in reduce_dims:
            num *= dim

        num_bw = 1.0 / num
        num_rec = tvm.const(num_bw, dtype="float32")

        if num == 1:
            batch_var_scalar = 0.0
        else:
            batch_var_scalar = float(num) / (num - 1)
    else:
        num_rec = tbe.var("num_rec", dtype="float32")
        batch_var_scalar = tbe.var("batch_var_scalar", dtype="float32")

    # compute the saved mean of x
    save_mean_reduce = tbe.vmuls(sum, num_rec)

    # compute the saved variance of x
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    # compute the oefficient of y
    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    # compute the batch normalization of x
    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = tbe.cast_to(x, "float32")

    res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")
    batch_variance = tbe.vmuls(save_variance_reduce, batch_var_scalar)

    mean_mul = tbe.vmuls(save_mean_reduce, factor)
    mean_mul_rev = tbe.vmuls(mean, factor_reverse)
    mean = tbe.vadd(mean_mul, mean_mul_rev)

    var_mul = tbe.vmuls(batch_variance, factor)
    var_mul_rev = tbe.vmuls(variance, factor_reverse)
    variance = tbe.vadd(var_mul, var_mul_rev)

    res = [res_y, mean, variance, save_mean_reduce, save_variance_reduce]

    return res


# 'pylint: disable=too-many-statements,too-many-arguments,too-many-locals,invalid-name,unused-argument
@register_operator("BNTrainingUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update(x,
                       sum,
                       square_sum,
                       scale,
                       offset,
                       mean,
                       variance,
                       y,
                       mean_out,
                       variance_out,
                       batch_mean,
                       batch_variance,
                       factor,
                       epsilon,
                       kernel_name="bn_training_update"):
    """
    algorithm: fused_batch_norm_v2
    Batch normalization.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A 5HD Tensor for sum.
        The output of batch_normalization_forward_training_reduce.
    square_sum: dict
        dict of square_sum, A 5HD Tensor for square_sum.
        The output of batch_normalization_forward_training_reduce.
    scale: dict
        dict of scale, A 5HD Tensor for scale.
    offset: dict
        dict of offset, A 5HD Tensor for offset.
    mean: dict
        dict of mean, A 5HD Tensor for mean.
    variance: dict
        dict of variance, A 5HD Tensor for variance.
    y: dict
        dict of output, A 'Tensor'. Has the same type as 'x'.
    mean_out: dict
        dict of mean, A 'Tensor'. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A 'Tensor'. The update variance of save variance and running variance.
    batch_mean: dict
        dict of batch_mean, A 'Tensor'.
        One of the results that is called save mean.
    batch_variance: dict
        dict of batch_variance, A 'Tensor'.
        Has the same type as 'batch_mean'.
    factor: float
        A ratio to calculate the update mean and variance.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_square_sum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    dtype_mean = mean.get("dtype").lower()
    dtype_variance = variance.get("dtype").lower()
    _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance)

    data_format = x.get("format")

    if is_unknown_rank_input((x, sum, square_sum, scale, offset, mean, variance)) or factor is None or epsilon is None:
        if data_format == "NC1HWC0":
            x["shape"] = [-1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        else:
            x["shape"] = [-1, -1, -1, -1, -1, 16]
            x["range"] = [(1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)]
            dynamic_shape = [1, 1, -1, 1, 1, 16]
            dynamic_range = [(1, 1), (1, 1), (1, None), (1, 1), (1, 1), (16, 16)]
        for input_dict in (sum, square_sum, scale, offset, mean, variance):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    shape_x = x.get("shape")
    reduce_shape = None
    dyn_flag = util_common.is_unknown([x, sum, square_sum, scale, offset, mean, variance])
    if not dyn_flag and data_format in ("NC1HWC0",):
        reduce_shape = [shape_x[0], shape_x[2], shape_x[3]]
    elif not dyn_flag and data_format in ("NDC1HWC0",):
        reduce_shape = [shape_x[0], shape_x[1], shape_x[3], shape_x[4]]

    ins_list = [x, sum, square_sum, scale, offset, mean, variance]
    ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedule_list, tensor_list = [], []

    for (ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset, ins_mean, ins_variance) in ins:
        with tbe.compute():
            _shape_x, _shape_sum, _shape_square_sum, _shape_scale, _shape_offset, _shape_mean, _shape_variance = \
                shape_util.variable_shape([ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset, ins_mean,
                                           ins_variance])

            x_input = tvm.placeholder(_shape_x, name="x_input", dtype=dtype_x)
            sum_input = tvm.placeholder(_shape_sum, name="sum_input", dtype=dtype_sum)
            square_sum_input = tvm.placeholder(_shape_square_sum, name="square_sum_input", dtype=dtype_square_sum)
            scale_input = tvm.placeholder(_shape_scale, name="scale_input", dtype=dtype_scale)
            offset_input = tvm.placeholder(_shape_offset, name="offset_input", dtype=dtype_offset)
            mean_input = tvm.placeholder(_shape_mean, name="mean_input", dtype=dtype_mean)
            variance_input = tvm.placeholder(_shape_variance, name="variance_input", dtype=dtype_variance)

            res = bn_training_update_compute(x_input,
                                             sum_input,
                                             square_sum_input,
                                             scale_input,
                                             offset_input,
                                             mean_input,
                                             variance_input,
                                             y,
                                             mean_out,
                                             variance_out,
                                             batch_mean,
                                             batch_variance,
                                             factor,
                                             epsilon,
                                             kernel_name=kernel_name,
                                             reduce_shape=reduce_shape,
                                             dyn_flag=dyn_flag)
            tensor_list.append(
                [x_input, sum_input, square_sum_input, scale_input, offset_input, mean_input, variance_input] +
                list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule_list.append(sch)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.build(schedule_list, config)
