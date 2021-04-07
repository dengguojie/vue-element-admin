# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
from impl.util.platform_adapter import tbe
from tbe import tvm
from tbe.common.utils import para_check, shape_util
from tbe.common.utils.errormgr import error_manager_vector
from impl.util.util_select_op_base import SplitInput, SplitOutput, get_op_cal_info
from tbe.dsl.base.operation import register_operator, get_context, add_compile_info, var
from impl.util.platform_adapter import register_operator_compute
from tbe.dsl.base.operation import var

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
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]], [1, [1], [-1], [-1]], [2, [1], [-1], [-1]], \
                                          [3, [1], [-1], [-1]], [4, [1], [-1], [-1]], [5, [1], [-1], [-1]], \
                                          [6, [1], [-1], [-1]]), \
                              SplitOutput([0, [1]], [1, [1]], [2, [1]], [3, [1]], [4, [1]])]]
    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset, shape_mean, shape_variance, format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    -----------
    shape_x: list or tuple
        x's data shape
    shape_sum: list or tuple
        sum's data shape
    shape_square_sum: list or tuple
        square_sum's data shape
    shape_scale: list or tuple
        scale's data shape
    shape_offset: list or tuple
        offset's data shape
    shape_mean: list or tuple
        mean's data shape
    shape_variance: list or tuple
        variance's data shape
    
    Returns
    -------
    None
    """
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_sum, param_name="sum")
    para_check.check_shape(shape_square_sum, param_name="square_sum")
    para_check.check_shape(shape_scale, param_name="scale")
    para_check.check_shape(shape_offset, param_name="offset")
    para_check.check_shape(shape_mean, param_name="mean")
    para_check.check_shape(shape_variance, param_name="variance")
    
    if len(shape_x) != 5 or len(shape_sum) != 5 or len(shape_square_sum) != 5 or \
            len(shape_scale) != 5:
        error_reson = "This operator can only support 5D, but some input's shape length is not 5"
        error_manager_vector.raise_err_specific_reson("bn_training_update", error_reson)
    if len(shape_offset) != 5 or len(shape_mean) != 5 or len(shape_variance) != 5 or \
            len(shape_scale) != 5:
        error_reson = "This operator can only support 5D, but some input's shape length is not 5"
        error_manager_vector.raise_err_specific_reson("bn_training_update", error_reson)
    dim_c1 = 0
    dim_c0 = 0
    i = 0
    j = 0
    if format == "NC1HWC0":
        dim_c1 = shape_x[1]
        dim_c0 = shape_x[4]
        i = 1
        j = 4
    if shape_sum[i] != dim_c1 or shape_sum[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and sum must be equal")
    if shape_square_sum[i] != dim_c1 or shape_square_sum[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", \
        "Dimension C of x and square_sum must be equal")
    if shape_scale[i] != dim_c1 or shape_scale[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and scale must be equal")
    if shape_offset[i] != dim_c1 or shape_offset[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and offset must be equal")
    if shape_mean[i] != dim_c1 or shape_mean[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", "Dimension C of x and mean must be equal")
    if shape_variance[i] != dim_c1 or shape_variance[j] != dim_c0:
        error_manager_vector.raise_err_specific_reson("bn_training_update", \
        "Dimension C of x and variance must be equal")


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


@register_operator_compute("BNTrainingUpdate", op_mode="dynamic", support_fusion=False)
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
                               kernel_name="bn_training_update"):  
    """
    algorithm: fused_batch_norm_v2
    Batch normalization

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: TVM tensor
        contains sum data
    square_sum:TVM tensor
        contains square_sum data
    scale:TVM tensor
        contains scale data
    offset:TVM tensor
        contains offset data
    mean:TVM tensor
        contains mean data
    variance:TVM tensor
        contains variance data
    y: dict
        dict of output, A 'Tensor'. Has the same tyupe as 'x'.
    mean_out: dict
        dict of mean, A 'Tensor'. The update mean of save mean and running mean.
    variance_out: dict
        dict of variance, A 'Tensor'. The update variance of save variance and running variance.
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
    
    Returns
    -------
    res: TVM tensor list
        the result of bn_training_update_compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    num = shape_x[0]*shape_x[2]*shape_x[3]

    if isinstance(num, int):
        num_rec = num ** (-1)
        num_rec = tvm.const(num_rec, dtype="float32")
        if num == 1:
            batch_var_scaler = 0.0
        else:
            batch_var_scaler = float(num) / (num - 1)
    else:
        num_rec = var("num_rec", dtype="float32")
        batch_var_scaler = var("batch_var_scaler", dtype="float32")
        add_compile_info("bn_update_num_rec_dtype", "float32")
        add_compile_info("bn_update_batch_var_scaler_dtype", "float32")
    
    save_mean_reduce = tbe.vmuls(sum, num_rec)
    variance_div = tbe.vmuls(square_sum, num_rec)
    variance_square = tbe.vmul(save_mean_reduce, save_mean_reduce)
    save_variance_reduce = tbe.vsub(variance_div, variance_square)

    multiplier_add = tbe.vadds(save_variance_reduce, epsilon)
    multiplier_sqrt = tbe.vsqrt(multiplier_add)
    multiplier_div = tbe.vdiv(scale, multiplier_sqrt)
    multiplier = tbe.broadcast(multiplier_div, shape_x)

    addend_mul = tbe.vmul(multiplier_div, save_mean_reduce)
    addend_sub = tbe.vsub(offset, addend_mul)
    addend = tbe.broadcast(addend_sub, shape_x)

    is_cast = False
    if x.dtype == "float16":
        is_cast = True
        x = tbe.cast_to(x, "float32")
    
    res_y = tbe.vadd(tbe.vmul(multiplier, x), addend)

    if is_cast:
        res_y = tbe.cast_to(res_y, "float16")
    batch_variance = tbe.vmuls(save_variance_reduce, batch_var_scaler)
    factor_reverse = 1.0 - factor
    mean_mul = tbe.vmuls(save_mean_reduce, factor)
    mean_mul_rev = tbe.vmuls(mean, factor_reverse)
    mean = tbe.vadd(mean_mul, mean_mul_rev)

    var_mul = tbe.vmuls(batch_variance, factor)
    var_mul_rev = tbe.vmuls(variance, factor_reverse)
    variance = tbe.vadd(var_mul, var_mul_rev)

    res = [res_y, mean, variance, save_mean_reduce, save_variance_reduce]

    return res


@register_operator("BNTrainingUpdate", "BnTrainingUpdate")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, 
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
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

    shape_x = x["shape"]
    shape_sum = sum["shape"]
    shape_square_sum = square_sum["shape"]
    shape_scale = scale["shape"]
    shape_offset = offset["shape"]
    shape_mean = mean["shape"]
    shape_variance = variance["shape"]

    dtype_x = x["dtype"].lower()
    dtype_sum = sum["dtype"].lower()
    dtype_square_sum = square_sum["dtype"].lower()
    dtype_scale = scale["dtype"].lower()
    dtype_offset = offset["dtype"].lower()
    dtype_mean = mean["dtype"].lower()
    dtype_variance = variance["dtype"].lower()

    format = x.get("format")

    _check_dtype(dtype_x, dtype_sum, dtype_square_sum, dtype_scale, dtype_offset, dtype_mean, dtype_variance)
    _check_shape(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset, shape_mean, shape_variance, format)

    ins = []
    ins_list = [x, sum, square_sum, scale, offset, mean, variance]
    for i in range(len(ins_list)):
        ins.append({'shape': ins_list[i]["shape"], 'ori_shape': ins_list[i]["shape"], 'mode': 'original'})
        shape_range = []
        for shape in ins_list[i]["shape"]:
            if shape == -1:
                shape_range.append((1, None))
            else:
                shape_range.append((shape, shape))
        ins[i]['range'] = tuple(shape_range)
    ins = [ins]

    schedule_list, tensor_list = [], []

    for (x, sum, square_sum, scale, offset, mean, variance) in ins:
        get_context().add("_support_broadcast", True)
        with tbe.compute():
            dim0_0 = var("dim0_0")
            dim0_1 = var("dim0_1")
            dim0_2 = var("dim0_2")
            dim0_3 = var("dim0_3")
            dim0_4 = var("dim0_4")

            shape_x = [dim0_0, dim0_1, dim0_2, dim0_3, dim0_4]
            shape_sum = [1, dim0_1, 1, 1, 16]

            x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x)
            sum_input = tvm.placeholder(shape_sum, name="sum_input", dtype=dtype_sum)
            square_sum_input = tvm.placeholder(shape_sum, name="square_sum_input", dtype=dtype_square_sum)
            scale_input = tvm.placeholder(shape_sum, name="scale_input", dtype=dtype_scale)
            offset_input = tvm.placeholder(shape_sum, name="offset_input", dtype=dtype_offset)
            mean_input = tvm.placeholder(shape_sum, name="mean_input", dtype=dtype_mean)
            variance_input = tvm.placeholder(shape_sum, name="variance_input", dtype=dtype_variance)

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
                                             kernel_name=kernel_name)
            tensor_list.append([x_input, sum_input, square_sum_input, scale_input, 
                                offset_input, mean_input, variance_input] + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule_list.append(sch)
    
    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.build(sch, config)
