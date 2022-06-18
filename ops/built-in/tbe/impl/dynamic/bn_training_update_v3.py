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
dynamic bn_training_update_v3
"""
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_compute import only_static_support
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import OpAttr


# 'pylint: disable=redefined-builtin
def _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset):
    """check input dtype"""
    para_check.check_dtype(dtype_x, ("float16", "float32"))
    para_check.check_dtype(dtype_sum, ("float32",))
    para_check.check_dtype(dtype_square_sum, ("float32",))
    para_check.check_dtype(dtype_scale, ("float32",))
    para_check.check_dtype(dtype_offset, ("float32",))


# 'pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
@register_operator_compute("BNTrainingUpdateV3", op_mode="dynamic", support_fusion=only_static_support)
def bn_training_update_v3_compute(x,
                                  sum,
                                  square_sum,
                                  scale,
                                  offset,
                                  y,
                                  batch_mean,
                                  batch_variance,
                                  reserve_1,
                                  reserve_2,
                                  epsilon,
                                  kernel_name="bn_training_update_v3",
                                  reduce_shape=None):
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
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"
    reduce_shape: list
        reduce shape of input shape

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_v3 compute
    """
    epsilon = get_attr_by_cls(epsilon, OpAttr(0, "epsilon", "Float", 0.0000001), "float32")

    shape_x = shape_util.shape_to_list(x.shape)
    data_format = y.get("format").upper()
    if not reduce_shape and data_format in ("NC1HWC0",) and len(shape_x) == 5:
        reduce_dims = [shape_x[0], shape_x[2], shape_x[3]]
    elif not reduce_shape and data_format in ("NDC1HWC0",) and len(shape_x) == 6:
        reduce_dims = [shape_x[0], shape_x[1], shape_x[3], shape_x[4]]
    else:
        reduce_dims = reduce_shape

    num = 1
    if reduce_dims:
        for dim in reduce_dims:
            num *= dim

    if reduce_dims and isinstance(num, int):
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
    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)
        res_y = tbe.cast_to(res_y, "float16")
    else:
        res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)

    # compute batch_mean and batch_var
    res_batch_mean = tbe.vmuls(sum, num_rec)
    res_batch_var = tbe.vmuls(save_variance_reduce, batch_var_scalar)

    res = [res_y, res_batch_mean, res_batch_var, save_mean_reduce, save_variance_reduce]

    return res


@register_operator("BNTrainingUpdateV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_update_v3(x,
                          sum,
                          square_sum,
                          scale,
                          offset,
                          y,
                          batch_mean,
                          batch_variance,
                          reserve_1,
                          reserve_2,
                          epsilon,
                          kernel_name="bn_training_update_v3"):
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
        dict of scale, A 5HD Tensor for mean.
    offset: dict
        dict of offset, A 5HD Tensor for variance.
    y: dict
        dict of output, A `Tensor`. Has the same type as `x`.
    batch_mean: dict
        dict of batch_mean, A `Tensor`.
        One of the result which is called save mean.
    batch_variance: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_1: dict
        dict of batch_mean, A `Tensor`.
        Has the same type as `batch_mean`.
    reserve_2: dict
        dict of batch_variance, A `Tensor`.
        Has the same type as `batch_variance`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_update_v3"

    Returns
    -------
    None
    """
    dtype_x = x.get("dtype").lower()
    dtype_sum = sum.get("dtype").lower()
    dtype_sqrsum = square_sum.get("dtype").lower()
    dtype_scale = scale.get("dtype").lower()
    dtype_offset = offset.get("dtype").lower()
    _check_dtype(dtype_x, dtype_sum, dtype_sqrsum, dtype_scale, dtype_offset)

    data_format = x.get("format").upper()
    if is_unknown_rank_input((x, sum, square_sum, scale, offset)) or epsilon is None:
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
        for input_dict in (sum, square_sum, scale, offset):
            input_dict["shape"] = dynamic_shape
            input_dict["range"] = dynamic_range

    shape_x = x.get("shape")
    reduce_shape = None
    dyn_flag = util_common.is_unknown([x, sum, square_sum, scale, offset])
    if not dyn_flag and data_format in ("NC1HWC0",):
        reduce_shape = [shape_x[0], shape_x[2], shape_x[3]]
    elif not dyn_flag and data_format in ("NDC1HWC0",):
        reduce_shape = [shape_x[0], shape_x[1], shape_x[3], shape_x[4]]

    ins_list = [x, sum, square_sum, scale, offset]
    ins = classify(ins_list, OpPatternMode.ELEWISE_WITH_BROADCAST)

    schedules = []
    tensors = []

    for (ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset) in ins:
        with tbe.compute():
            _shape_x, _shape_sum, _shape_sqrsum, _shape_scale, _shape_offset = shape_util.variable_shape(
                [ins_x, ins_sum, ins_square_sum, ins_scale, ins_offset])

            in_x = tvm.placeholder(_shape_x, name="x", dtype=dtype_x)
            in_sum = tvm.placeholder(_shape_sum, name="sum", dtype=dtype_sum)
            in_sqrsum = tvm.placeholder(_shape_sqrsum, name="sqrsum", dtype=dtype_sum)
            in_scale = tvm.placeholder(_shape_scale, name="scale", dtype=dtype_sum)
            in_offset = tvm.placeholder(_shape_offset, name="offset", dtype=dtype_sum)
            res = bn_training_update_v3_compute(in_x,
                                                in_sum,
                                                in_sqrsum,
                                                in_scale,
                                                in_offset,
                                                y,
                                                batch_mean,
                                                batch_variance,
                                                reserve_1,
                                                reserve_2,
                                                epsilon,
                                                kernel_name=kernel_name,
                                                reduce_shape=reduce_shape)
            tensors.append([in_x, in_sum, in_sqrsum, in_scale, in_offset] + res)

            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
