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
bn_training_reduce_grad
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable = unused-argument
# 'pylint: disable=invalid-name,too-many-arguments,consider-using-in
def get_op_support_info(grads, x, diff_scale, diff_offset, scale,
                        batch_mean, batch_variance, y, epsilon=0.0001,
                        kernel_name="bn_training_reduce_grad"):
    """
    get_op_support_info
    """
    format_grads = grads.get("format").upper()
    if format_grads == "NC1HWC0" or format_grads == "NCHW":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]], [1, [1], [-1], [-1]], [2, [0], [-1], [-1]], \
                                         [3, [0], [-1], [-1]], [4, [0], [-1], [-1]], [5, [0], [-1], [-1]], \
                                         [6, [0], [-1], [-1]]), SplitOutput([0, [1]])]]

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,invalid-name,too-many-arguments
# 'pylint: disable=locally-disabled,too-many-statements,unused-argument
# 'pylint: disable=locally-disabled,too-many-locals
def op_select_format(grads, x, diff_scale, diff_offset, scale,
                     batch_mean, batch_variance, y, epsilon,
                     kernel_name="bn_training_reduce_grad"):
    """
    1. when input(grads)'s ori_shape is [1, ? ,1, ?] and the format is NCHW
    the Op BNTrainingReduceGrad can support NCHW.
    > for example:
    > grads : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > x : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > diff_scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > diff_offset : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > scale : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_mean : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > batch_variance : Tensor of (shape=(1, 16, 1, 16), "NCHW")
    > the Op BNTrainingReduce can process with NC1HWC0:
    > grads : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > x : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > diff_scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > diff_offset : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > scale : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_mean : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    > batch_variance : Tensor of (shape=(1, 16, 1, 2, 8), "NC1HWC0")
    """
    format_grads = grads.get("ori_format").upper()
    origin_shape = grads.get("ori_shape")

    # can support ND + ND
    if format_grads == "NCHW" and len(origin_shape) == 4 \
            and origin_shape[0] == 1 and origin_shape[2] == 1:
        input0 = util_select_op_base.gen_param(classify="input0", name="grads",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1", name="x",
                                               datatype="float16,float,float16,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2", name="diff_scale",
                                               datatype="float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3", name="diff_offset",
                                               datatype="float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input4 = util_select_op_base.gen_param(classify="input4", name="scale",
                                               datatype="float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input5 = util_select_op_base.gen_param(classify="input5", name="batch_mean",
                                               datatype="float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        input6 = util_select_op_base.gen_param(classify="input6", name="batch_variance",
                                               datatype="float,float,float,float",
                                               format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float,float16,float",
                                                format="NCHW,NCHW,NC1HWC0,NC1HWC0",
                                               unknownshape_format="NCHW,NCHW,NC1HWC0,NC1HWC0")
    # support 5HD + 5HD
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="grads",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input1 = util_select_op_base.gen_param(classify="input1", name="x",
                                               datatype="float16,float,float16,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input2 = util_select_op_base.gen_param(classify="input2", name="diff_scale",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input3 = util_select_op_base.gen_param(classify="input3", name="diff_offset",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input4 = util_select_op_base.gen_param(classify="input4", name="scale",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input5 = util_select_op_base.gen_param(classify="input5", name="batch_mean",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        input6 = util_select_op_base.gen_param(classify="input6", name="batch_variance",
                                               datatype="float,float,float,float",
                                               format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                               unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")
        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                datatype="float16,float,float16,float",
                                                format="NC1HWC0,NC1HWC0,NDC1HWC0,NDC1HWC0",
                                                unknownshape_format="NC1HWC0, NC1HWC0, NDC1HWC0, NDC1HWC0")

    param_list = [input0, input1, input2, input3,
                  input4, input5, input6, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def _check_format_nd(data_format, origin_foramt):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    data_format: str
        data format of data
    origin_foramt: str
        origin format of data

    Returns
    -------
    None
    """
    if data_format.upper() not in ("NC1HWC0", "NCHW", "NDC1HWC0"):
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                      "The data format only supports NC1HWC0,NDC1HWC0,and NCHW.")
    if data_format.upper() == "NCHW":
        if origin_foramt not in ("NCHW",):
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad",
                                                          "The origin format only supports NCHW when format is NCHW")


@tbe_platform.fusion_manager.fusion_manager.register("bn_training_reduce_grad")
def bn_training_reduce_grad_compute(grads, x, diff_scale, diff_offset, scale,
                                    batch_mean, batch_variance, y, epsilon,
                                    kernel_name="bn_training_reduce_grad"):
    """
    Compute for batch_norm_train_reduce_grad
    y:(grads*scale*np.power((batch_variance + epsilon), (-0.5)))+
      np.sum(grads*scale*(-0.5)*x_norm*np.power((batch_variance+epsilon),(-1))))
      *(2/m)+np.sum(grads*scale*(-1)*
      np.power((batch_variance+epsilon),(-0.5)))*(1/m)

    Parameters
    ----------
    grads: TVM tensor 5D
        the placeholder of grads.
        Must be one of the following type: `float16`, `float32`.
    x: TVM tensor 5D
        the placeholder of x.
        Must be one of the following type: `float32`, 'float16.
    diff_scale: TVM tensor 5D
        the placeholder of diff_scale.
        Must be one of the following type: `float32`.
    diff_offset: TVM tensor 5D
         the placeholder of diff_offset.
         Must be one of the following types: `float32`.
    scale: TVM tensor 5D
        the placeholder of scale.
        Must be one of the following types: `float32`.
    batch_mean: dict 5D
        the placeholder of batch_mean.
        Must be one of the following types: `float32`.
    batch_variance: dict 5D
        the placeholder of batch_variance.
        Must be one of the following types: `float32`.
    y: dict
        dict of y, include keys(shape and dtype).
    epsilon: float
        A small float number added to the variance of x.

    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    res: TVM tensor
    """
    shape_grads = shape_util.shape_to_list(grads.shape)
    num = shape_grads[0] * shape_grads[2] * shape_grads[3]
    num_rec = 1.0 / num
    is_cast = False
    if grads.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        is_cast = True
        grads = tbe.cast_to(grads, "float32")

    if x.dtype == "float16" and \
            tbe_platform.cce_conf.api_check_support("te.lang.cce.vdiv", "float32"):
        x = tbe.cast_to(x, "float32")

    data_sqrt = tbe.vsqrt(tbe.vadds(batch_variance, epsilon))
    scale_inv = tbe.vmuls(diff_scale, num_rec)
    scale_inv_reverse = tbe.vmuls(diff_scale, (-1.0)*num_rec)
    offset_inv_reverse = tbe.vmuls(diff_offset, (-1.0)*num_rec)

    multiplier = tbe.vdiv(scale_inv_reverse, data_sqrt)
    addend_div = tbe.vdiv(batch_mean, data_sqrt)
    addend_mul = tbe.vmul(addend_div, scale_inv)
    addend = tbe.vadd(addend_mul, offset_inv_reverse)

    multiplier_broadcast = tbe.broadcast(multiplier, shape_grads)
    addend_broadcast = tbe.broadcast(addend, shape_grads)

    coef_mul = tbe.vmul(multiplier_broadcast, x)
    coef_add = tbe.vadd(grads, coef_mul)
    coef = tbe.vadd(coef_add, addend_broadcast)

    mul_scale = tbe.vdiv(scale, data_sqrt)
    mul_scale_broadcast = tbe.broadcast(mul_scale, shape_grads)

    res = tbe.vmul(coef, mul_scale_broadcast)

    if is_cast:
        res = tbe.cast_to(res, "float16")
    return res


def _check_shape(shape_grads, shape_diff_scale, data_format):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_grads: list or tuple
        input grads's data shape
    shape_diff_scale: list or tuple
        input diff_scale's data shape
    Returns
    -------
    None
    """
    para_check.check_shape(shape_grads, param_name="grads")
    para_check.check_shape(shape_diff_scale, param_name="diff_scale")
    dim_c0 = 0
    dim_c1 = 0
    if data_format == "NDC1HWC0":
        dim_c1 = shape_grads[2]
        dim_c0 = shape_grads[5]
        n_shape = shape_diff_scale[0] * shape_diff_scale[1]
        if n_shape != 1 or shape_diff_scale[3] != 1 or shape_diff_scale[4] != 1:
            error_reson = "Dimensions except Dimension C must be one for shape_diff_scale"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
        if shape_diff_scale[2] != dim_c1 or shape_diff_scale[5] != dim_c0:
            error_reson = "Dimension C must be equal"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
    else:
        dim_c1 = shape_grads[1]
        dim_c0 = shape_grads[4]
        if shape_diff_scale[0] != 1 or shape_diff_scale[2] != 1 or shape_diff_scale[3] != 1:
            error_reson = "Dimensions except Dimension C must be one for shape_diff_scale"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
        if shape_diff_scale[1] != dim_c1 or shape_diff_scale[4] != dim_c0:
            error_reson = "Dimension C must be equal"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)

    if len(shape_grads) not in (5, 6) or len(shape_diff_scale) not in (5, 6):
        error_reson = "This operator can only support 5D,6D, but some input's shape length is not 5 or 6"
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)
    if dim_c0 != 16:
        error_reson = "shape_grads last dim must be 16"
        error_manager_vector.raise_err_specific_reson("bn_training_reduce_grad", error_reson)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def bn_training_reduce_grad(grads, x, diff_scale, diff_offset, scale,
                            batch_mean, batch_variance, y, epsilon=0.0001,
                            kernel_name="bn_training_reduce_grad"):
    """
    algorithm: fused_batch_norm_grad_v2
    bn_training_reduce_grad.

    Parameters
    ----------
    grads: dict
        dict of grads, A 5D Tensor for input grads.
        source data type, support "float32", "float16".
    x: dict
        dict of s, A 5D Tensor for input x.
        source data type, support "float32", "float16".
    diff_scale: dict
        dict of diff_scale, A 5D Tensor for input diff_scale.
        The output of bn_training_update_grad.
        source data type, support "float32".
    diff_offset: dict
        dict of diff_offset, A 5HD Tensor for input diff_offset.
        The output of bn_training_update_grad.
        source data type, support "float32".
    scale: dict
        dict of scale, A 5HD Tensor for input scale.
        source data type, support "float32".
    batch_mean: dict
        dict of batch_mean, A 5D Tensor for input batch_mean.
        source data type, support "float32".
    batch_variance: dict
        dict of batch_variance, A 5D Tensor for input batch_variance.
        source data type, support "float32".
    y: dict
        dict of output, A `Tensor`. Has the same type as `grads`.
    epsilon: float
        A small float number added to the variance of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce_grad"

    Returns
    -------
    None
    """

    shape_grads = grads.get("shape")
    shape_x = x.get("shape")
    shape_diff_scale = diff_scale.get("shape")
    shape_scale = scale.get("shape")
    shape_util.compare_tensor_dict_key(grads, x, "shape")

    dtype_grads = grads.get("dtype")
    dtype_x = x.get("dtype")
    dtype_diff_scale = diff_scale.get("dtype")
    dtype_diff_offset = diff_offset.get("dtype")
    dtype_scale = scale.get("dtype")
    dtype_batch_mean = batch_mean.get("dtype")
    dtype_batch_variance = batch_variance.get("dtype")

    input_grads_dtype = dtype_grads.lower()
    x_dtype = dtype_x.lower()
    diff_scale_dtype = dtype_diff_scale.lower()
    diff_offset_dtype = dtype_diff_offset.lower()
    scale_dtype = dtype_scale.lower()
    batch_mean_dtype = dtype_batch_mean.lower()
    batch_variance_dtype = dtype_batch_variance.lower()

    para_check.check_dtype(input_grads_dtype, ("float32", "float16"), param_name="grads")
    para_check.check_dtype(x_dtype, ("float32", "float16"), param_name="x")
    para_check.check_dtype(diff_scale_dtype, ("float32",), param_name="diff_scale")
    para_check.check_dtype(diff_offset_dtype, ("float32",), param_name="diff_offset")
    para_check.check_dtype(scale_dtype, ("float32",), param_name="scale")
    para_check.check_dtype(batch_mean_dtype, ("float32",), param_name="batch_mean")
    para_check.check_dtype(batch_variance_dtype, ("float32",), param_name="batch_variance")

    shape_util.compare_tensor_dict_key(diff_scale, diff_offset, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, scale, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, batch_mean, "shape")
    shape_util.compare_tensor_dict_key(diff_scale, batch_variance, "shape")
    shape_util.compare_tensor_dict_key(grads, x, "shape")

    data_format = grads.get("format").upper()
    ori_format = grads.get("ori_format").upper()
    _check_format_nd(data_format, ori_format)

    if data_format in ("NC1HWC0", "NDC1HWC0"):
        _check_shape(shape_grads, shape_diff_scale, data_format)
    else:
        shape_list = [1, 1, 1, 1]
        shape_list[1] = shape_x[1]
        shape_diff_scale = shape_list
        shape_scale = shape_list
    if data_format == "NDC1HWC0":
        shape_grads = [shape_grads[0] * shape_grads[1], shape_grads[2], shape_grads[3], shape_grads[4], shape_grads[5]]
        shape_scale = [shape_scale[0] * shape_scale[1], shape_scale[2], shape_scale[3], shape_scale[4], shape_scale[5]]
    grads_input = tvm.placeholder(shape_grads, name="grads_input",
                                  dtype=input_grads_dtype)
    x_input = tvm.placeholder(shape_grads, name="x_input", dtype=x_dtype)
    diff_scale_input = tvm.placeholder(shape_scale,
                                       name="diff_scale_input",
                                       dtype=diff_scale_dtype)
    diff_offset_input = tvm.placeholder(shape_scale,
                                        name="diff_offset_input",
                                        dtype=diff_offset_dtype)
    scale_input = tvm.placeholder(shape_scale, name="scale_input",
                                  dtype=scale_dtype)
    batch_mean_input = tvm.placeholder(shape_scale,
                                       name="batch_mean_input",
                                       dtype=batch_mean_dtype)
    batch_variance_input = tvm.placeholder(shape_scale,
                                           name="batch_variance_input",
                                           dtype=batch_variance_dtype)

    res = bn_training_reduce_grad_compute(grads_input, x_input,
                                          diff_scale_input, diff_offset_input,
                                          scale_input, batch_mean_input,
                                          batch_variance_input, y, epsilon,
                                          kernel_name=kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    tensor_list = [grads_input, x_input, diff_scale_input, diff_offset_input,
                   scale_input, batch_mean_input, batch_variance_input, res]
    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
