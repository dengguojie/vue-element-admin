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
layer_norm
"""
from te import tvm
from te import platform as tbe_platform
import te.lang.cce as tbe
from te.utils import shape_util
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.layer_norm_tik import if_tik_support
from impl.layer_norm_tik import layer_normalize

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648

SIZE_SIXTEEN = 16


# pylint: disable = unused-argument
# pylint: disable=too-many-arguments,too-many-locals
def get_op_support_info(input_x, input_gamma, input_beta,
                        output_y, output_mean, output_variance,
                        begin_norm_axis, begin_params_axis,
                        epsilon=1e-12, kernel_name="layer_norm",
                        impl_mode="high_performance"):
    """
    get_op_support_info
    """
    format_x = input_x.get("format").upper()
    shape_x = input_x.get("shape")
    begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
    begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)
    if format_x in ("ND", "NCHW", "NHWC", "NC1HWC0"):
        axis_split_matrix=[]
        if begin_params_axis == 0:
            for i in range(begin_norm_axis):
                split_0 = [SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]]), \
                           SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                axis_split_matrix.append(split_0)
        else:
            if begin_norm_axis <= begin_params_axis:
                for i in range(begin_norm_axis):
                    split_0 = [SplitInput([0, [i], [-1], [-1]]), \
                               SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                    axis_split_matrix.append(split_0)
            else:
                for i in range(begin_params_axis):
                    split_0 = [SplitInput([0, [i], [-1], [-1]]), \
                               SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                    axis_split_matrix.append(split_0)

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=locally-disabled,too-many-arguments,unused-argument
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def _division_sixteen(shape):

    if len(shape) < 2:
        if shape[-1] == 0:
            error_detail = "value of shape_x is illegal"
            error_manager_vector.raise_err_input_shape_invalid("layer_norm", "input_x", \
                                                               error_detail)
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = "value of shape_x is illegal"
        error_manager_vector.raise_err_input_shape_invalid("layer_norm", "input_x", \
                                                           error_detail)

    if shape[-1] % SIZE_SIXTEEN == 0 and shape[-2] % SIZE_SIXTEEN == 0:
        return True
    return False


def op_select_format(input_x, input_gamma, input_beta,
                     output_y, output_mean, output_variance,
                     begin_norm_axis, begin_params_axis,
                     kernel_name="layer_norm"):
    """
    select format dynamically
    """
    shape_x = input_x.get("ori_shape")
    shape_x = shape_util.scalar2tensor_one(shape_x)
    shape_gamma = input_gamma.get("ori_shape")
    shape_gamma = shape_util.scalar2tensor_one(shape_gamma)

    # can not support Nz + ND
    # while len(shape_gamma) >= 2 and  _division_sixteen(shape_x) = False
    if begin_params_axis == 0:
        if len(shape_gamma) >= 2 or (not _division_sixteen(shape_x)):
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float",
                                                   format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1", name="gamma",
                                                   datatype="float16,float16,float16,float16,float,"
                                                            "float,float,float",
                                                   format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2", name="beta",
                                                   datatype="float16,float16,float16,float16,float,"
                                                            "float,float,float",
                                                   format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,float16,float16,float,"
                                                             "float,float,float",
                                                    format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1", name="mean",
                                                    datatype="float16,float16,float16,float16,float,"
                                                             "float,float,float",
                                                    format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2", name="variance",
                                                    datatype="float16,float16,float16,float16,float,"
                                                             "float,float,float",
                                                    format="NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,NHWC,ND")
        else:
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float,float16,float16,float16,"
                                                            "float16,float,float,float,float",
                                                   format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NC1HWC0,NHWC,"
                                                          "ND,NCHW,NC1HWC0,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1", name="gamma",
                                                   datatype="float16,float,float16,float16,float16,"
                                                            "float16,float,float,float,float",
                                                   format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                                          "NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2", name="beta",
                                                   datatype="float16,float,float16,float16,float16,"
                                                            "float16,float,float,float,float",
                                                   format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                                          "NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float,float16,float16,float16,"
                                                             "float16,float,float,float,float",
                                                    format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1", name="mean",
                                                    datatype="float16,float,float16,float16,float16,"
                                                             "float16,float,float,float,float",
                                                    format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                                           "NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2", name="variance",
                                                    datatype="float16,float,float16,float16,float16,"
                                                             "float16,float,float,float,float",
                                                    format="ND,ND,NCHW,NC1HWC0,NHWC,ND,NCHW,NC1HWC0,"
                                                           "NHWC,ND")
    else:
        if len(shape_gamma) >= 2 or (not _division_sixteen(shape_x)):
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float",
                                                   format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1", name="gamma",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float",
                                                   format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2", name="beta",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float",
                                                   format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float",
                                                    format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1", name="mean",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float",
                                                    format="NCHW,NHWC,ND,NCHW,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2", name="variance",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float",
                                                    format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        else:
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float,float16,float16,"
                                                            "float16,float,float,float",
                                                   format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                          "ND,NCHW,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1", name="gamma",
                                                   datatype="float16,float,float16,float16,"
                                                            "float16,float,float,float",
                                                   format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                          "NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2", name="beta",
                                                   datatype="float16,float,float16,float16,"
                                                            "float16,float,float,float",
                                                   format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                          "NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float,float16,float16,"
                                                             "float16,float,float,float",
                                                    format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1", name="mean",
                                                    datatype="float16,float,float16,float16,"
                                                             "float16,float,float,float",
                                                    format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                           "NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2", name="variance",
                                                    datatype="float16,float,float16,float16,"
                                                             "float16,float,float,float",
                                                    format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                           "NHWC,ND")

    param_list = [input0, input1, input2, output0, output1, output2]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """

    frac_z_axis = list(ori_axis)
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] > shape_len - 2:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 1)
            else:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


def layer_norm_compute_nz(input_x, input_gamma, input_beta,
                          output_y, output_mean, output_variance,
                          begin_norm_axis, begin_params_axis,
                          ori_shape, epsilon, kernel_name="layer_norm",
                          impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    cast_dtype = "float16"
    cast_dtype_precision = dtype
    if dtype == "float16" and \
            ((tbe_platform.cce_conf.api_check_support
                  ("te.lang.cce.vexp", "float32") and
              impl_mode == "high_performance") or
             impl_mode == "high_precision"):
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")

    # Calculate the scaling ratio of the average
    reduce_elts = 1.0
    index_list = tuple(index for index, _ in enumerate(ori_shape))
    reduce_axis = index_list[begin_norm_axis:]
    for i in reduce_axis:
        reduce_elts *= ori_shape[i]
    reduce_axis = to_frac_z_axis(ori_shape, reduce_axis)
    mean_cof = reduce_elts ** (-1)
    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(input_x, mean_cof)
    mean = tbe.sum(mean_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the variance calculation process
    mean_variance_broadcast = _broadcast_nz(mean, shape_x)
    variance_sub = tbe.vsub(input_x, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    variance = tbe.sum(variance_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the normalize calculation process
    if impl_mode == "high_performance":
        mean_normalize_broadcast = _broadcast_nz(mean, shape_x)
        normalize_sub = tbe.vsub(input_x, mean_normalize_broadcast)
        epsilon = tvm.const(epsilon, dtype=cast_dtype)
        variance_normalize_broadcast = _broadcast_nz(variance, shape_x)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = \
            tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
        normalize_exp = tbe.vexp(normalize_log_mul)
        normalize_mul = tbe.vmul(normalize_sub, normalize_exp)
    else:
        tesor_one = tbe.broadcast(tvm.const
                                  (1, cast_dtype_precision),
                                  shape_x)
        mean_normalize_broadcast = _broadcast_nz(mean, shape_x)
        normalize_sub = tbe.vsub(input_x, mean_normalize_broadcast)
        variance_normalize_broadcast = _broadcast_nz(variance, shape_x)
        epsilon = tvm.const(epsilon, dtype=cast_dtype_precision)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_sqrt = tbe.vsqrt(normalize_add, 0)
        normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
        normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(input_gamma, normalize_mul)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = _broadcast_nz(input_gamma, shape_x)
        beta_broadcast = _broadcast_nz(input_beta, shape_x)
        scale_mul = tbe.vmul(gamma_broadcast, normalize_mul)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if dtype == "float16" and \
            ((tbe_platform.cce_conf.api_check_support
                  ("te.lang.cce.vexp", "float32") and
              impl_mode == "high_performance") or
             impl_mode == "high_precision"):
        mean = tbe.cast_to(mean, "float16")
        variance = tbe.cast_to(variance, "float16")
        res = tbe.cast_to(res, "float16")

    return mean, variance, res


@tbe_platform.fusion_manager.fusion_manager.register("layer_norm")
def layer_norm_compute(input_x, input_gamma, input_beta,
                       output_y, output_mean, output_variance,
                       begin_norm_axis, begin_params_axis,
                       epsilon, kernel_name="layer_norm",
                       impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_data: dict
        shape and dtype of output
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "cce_layernorm"

    Returns
    -------
    res_tuple: tuple
        (mean, variance, result)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    cast_dtype = "float16"
    cast_dtype_precision = dtype
    if dtype == "float16" and \
            ((tbe_platform.cce_conf.api_check_support
                  ("te.lang.cce.vexp", "float32") and
              impl_mode == "high_performance") or
             impl_mode == "high_precision"):
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")

    # Calculate the scaling ratio of the average
    index_list = tuple(index for index, _ in enumerate(shape_x))
    reduce_axis = index_list[begin_norm_axis:]

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    mean_cof = reduce_elts ** (-1)

    # DSL description of the mean calculation process
    mean_muls = tbe.vmuls(input_x, mean_cof)
    mean = tbe.sum(mean_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the variance calculation process
    mean_variance_broadcast = tbe.broadcast(mean, shape_x)
    variance_sub = tbe.vsub(input_x, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    variance_muls = tbe.vmuls(variance_mul, mean_cof)
    variance = tbe.sum(variance_muls, axis=reduce_axis, keepdims=True)

    # DSL description of the normalize calculation process
    if impl_mode == "high_performance":
        mean_normalize_broadcast = tbe.broadcast(mean, shape_x)
        normalize_sub = tbe.vsub(input_x, mean_normalize_broadcast)
        epsilon = tvm.const(epsilon, dtype=cast_dtype)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = \
            tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
        normalize_exp = tbe.vexp(normalize_log_mul)
        normalize_mul = tbe.vmul(normalize_sub, normalize_exp)
    else:
        tesor_one = tbe.broadcast(tvm.const
                                  (1, cast_dtype_precision),
                                  shape_x)
        mean_normalize_broadcast = tbe.broadcast(mean, shape_x)
        normalize_sub = tbe.vsub(input_x, mean_normalize_broadcast)
        variance_normalize_broadcast = tbe.broadcast(variance, shape_x)
        epsilon = tvm.const(epsilon, dtype=cast_dtype_precision)
        normalize_add = tbe.vadds(variance_normalize_broadcast, epsilon)
        normalize_sqrt = tbe.vsqrt(normalize_add, 0)
        normalize_rsqrt = tbe.vdiv(tesor_one, normalize_sqrt)
        normalize_mul = tbe.vmul(normalize_sub, normalize_rsqrt)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(input_gamma, normalize_mul)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = tbe.broadcast(input_gamma, shape_x)
        beta_broadcast = tbe.broadcast(input_beta, shape_x)
        scale_mul = tbe.vmul(gamma_broadcast, normalize_mul)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if dtype == "float16" and \
            ((tbe_platform.cce_conf.api_check_support
                  ("te.lang.cce.vexp", "float32") and
              impl_mode == "high_performance") or
             impl_mode == "high_precision"):
        mean = tbe.cast_to(mean, "float16")
        variance = tbe.cast_to(variance, "float16")
        res = tbe.cast_to(res, "float16")

    return mean, variance, res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def layer_norm(input_x, input_gamma, input_beta,
               output_y, output_mean, output_variance,
               begin_norm_axis, begin_params_axis,
               epsilon=1e-12, kernel_name="layer_norm",
               impl_mode="high_performance"):
    """
    layernorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, reduce_axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    input_beta: dict
        shape and dtype of input beta, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layernorm"

    Returns
    -------
    None
    """
    shape_x = list(input_x.get("shape"))
    input_gamma_shape = input_gamma.get("shape")
    input_beta_shape = input_beta.get("shape")
    ori_shape_x = list(input_x.get("ori_shape"))
    input_format = input_x.get("format").upper()
    input_gamma_format = input_gamma.get("format").upper()
    input_beta_format = input_beta.get("format").upper()

    para_check.check_shape(input_gamma_shape, param_name="input_gamma")
    para_check.check_shape(input_beta_shape, param_name="input_beta")
    para_check.check_shape(shape_x, param_name="input_x")

    check_list = ("float16", "float32")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_gamma.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="input_gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="input_gamma")

    shape_gamma = list(input_gamma.get("shape"))
    shape_beta = list(input_beta.get("shape"))

    tik_support = if_tik_support(input_x, input_gamma, input_beta, output_y, output_mean,
                                 output_variance, begin_norm_axis, begin_params_axis, epsilon)
    if tik_support:
        layer_normalize(input_x, input_gamma, input_beta,
                        output_y, output_mean, output_variance,
                        begin_norm_axis, begin_params_axis,
                        epsilon, kernel_name)
    else:
        if input_format == "FRACTAL_NZ":
            begin_norm_axis = shape_util.axis_check(len(ori_shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(ori_shape_x), begin_params_axis)

            if input_gamma_format == "FRACTAL_NZ" or \
                    input_beta_format == "FRACTAL_NZ":
                error_detail = "gamma and beta not support Nz in bert"
                error_manager_vector.raise_err_two_input_format_invalid(kernel_name, "input_gamma", \
                                                                        "input_beta", error_detail)
            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", \
                                                                       "input_beta", error_detail)
            if ori_shape_x[begin_params_axis:] != shape_gamma:
                error_detail = "x or gamma or begin_params_axis is wrong."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", \
                                                                       "input_gamma", error_detail)
            if len(shape_gamma) > 1:
                error_detail = "shape of gamma or beta only support 1D in bert"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_gamma", error_detail)

            # make shape_x,shape_gamma,shape_beta dim same
            if begin_params_axis != 0:
                for i in range(begin_params_axis):
                    shape_gamma.insert(i, 1)
            shape_gamma[-2] = shape_x[-4]
            shape_gamma[-1] = 1
            shape_gamma.append(1)
            shape_gamma.append(shape_x[-1])
            if begin_params_axis > len(ori_shape_x) - 2:
                shape_x[-3:] = [shape_x[-3]*shape_x[-2], shape_x[-1]]
                shape_gamma[-3:] = [shape_gamma[-3]*shape_gamma[-2], shape_gamma[-1]]
            shape_beta = shape_gamma
        else:
            begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)

            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", \
                                                                       "input_beta", error_detail)
            no_need_fix_gamma = False
            no_need_fix_beta = False
            if shape_x[begin_params_axis:] != shape_gamma:
                if len(shape_x) == len(shape_gamma):
                    no_need_fix_gamma = True
                else:
                    error_detail = "x or gamma or begin_params_axis is wrong."
                    error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", \
                                                                           "input_gamma", error_detail)
            if shape_x[begin_params_axis:] != shape_beta:
                if len(shape_x) == len(shape_beta):
                    no_need_fix_beta = True
                else:
                    error_detail = "x or gamma or begin_params_axis is wrong."
                    error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", \
                                                                           "input_beta", error_detail)
            # make shape_x,shape_gamma,shape_beta dim same
            if begin_params_axis != 0 and not no_need_fix_gamma:
                for i in range(begin_params_axis):
                    shape_gamma.insert(i, 1)
            if begin_params_axis != 0 and not no_need_fix_beta:
                for i in range(begin_params_axis):
                    shape_beta.insert(i, 1)

        data_x = tvm.placeholder(shape_x, name="x", dtype=dtype)
        data_gamma = tvm.placeholder(shape_gamma, name="gamma", dtype=dtype)
        data_beta = tvm.placeholder(shape_beta, name="beta", dtype=dtype)

        if input_format == "FRACTAL_NZ":

            mean, variance, res = \
                layer_norm_compute_nz(data_x, data_gamma, data_beta,
                                      output_y, output_mean, output_variance,
                                      begin_norm_axis, begin_params_axis,
                                      ori_shape_x, epsilon, kernel_name, impl_mode)
        else:

            mean, variance, res = \
                layer_norm_compute(data_x, data_gamma, data_beta,
                                   output_y, output_mean,
                                   output_variance,
                                   begin_norm_axis, begin_params_axis,
                                   epsilon, kernel_name, impl_mode)

        with tvm.target.cce():
            sch = tbe.auto_schedule([res, mean, variance])

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": [data_x, data_gamma,
                                  data_beta, res, mean, variance]}

        tbe.cce_build_code(sch, config)
