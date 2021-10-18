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
layer_norm_beta_gamma_backprop_v2
"""
import operator
import te.lang.cce as tbe
import functools

from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# pylint: disable = unused-argument,invalid-name,too-many-locals,too-many-arguments,too-many-branches
def get_op_support_info(input_dy, input_x, output_pd_gamma,
                        output_pd_beta, shape_gamma,
                        kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    get_op_support_info
    """
    shape_x = input_x.get("shape")
    format_dy = input_dy.get("format").upper()
    if format_dy in ("ND", "NCHW", "NHWC", "NC1HWC0"):
        if len(shape_x) == len(shape_gamma):
            axis_split_matrix = []
            for i in range(len(shape_x)):
                split_0 = [SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]]),
                           SplitOutput([0, [i]], [1, [i]])]
                axis_split_matrix.append(split_0)
        else:
            axis_split_matrix = None

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# pylint: disable=locally-disabled,unused-argument,too-many-arguments
# pylint: disable=too-many-locals,too-many-lines,too-many-function-args
def _check_dynamic_format(shape_dy, shape_gamma, c_0):
    """
    check dynamic format branch

    """
    if len(shape_dy) < 2 or len(shape_gamma) != 1:
        return True
    if shape_dy[-1] % c_0 != 0 or shape_dy[-2] % c_0 != 0 \
            or shape_gamma[-1] % c_0 != 0:
        return True
    return False


def op_select_format(input_dy, input_x,
                     output_pd_gamma, output_pd_beta, shape_gamma,
                     kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    function of selecting dynamic format

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output, only support float16, float32
    shape_gamma: tuple or list
        shape input gamma, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    None
    """
    shape_dy = input_dy.get("ori_shape")
    shape_dy = shape_util.scalar2tensor_one(shape_dy)
    c_0 = 16

    if _check_dynamic_format(shape_dy, shape_gamma, c_0):
        input0 = util_select_op_base.gen_param(classify="input0", name="dy",
                                               datatype="float,float,float",
                                               format="NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="res_for_gamma",
                                               datatype="float,float,float",
                                               format="NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="pd_gamma",
                                                datatype="float,float,float",
                                                format="NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="pd_beta",
                                                datatype="float,float,float",
                                                format="NCHW,NHWC,ND")
    else:
        input0 = util_select_op_base.gen_param(classify="input0", name="dy",
                                               datatype="float16,float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                      "ND,NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1", name="res_for_gamma",
                                               datatype="float,float,float,float,"
                                                        "float,float,float,float",
                                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                      "ND,NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0", name="pd_gamma",
                                                datatype="float,float,float,float,"
                                                         "float,float,float,float",
                                                format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                       "NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1", name="pd_beta",
                                                datatype="float,float,float,float,float,"
                                                         "float,float,float",
                                                format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                       "NHWC,ND")

    param_list = [input0, input1, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_params(params_map):
    """
    check parameters including shape_dy, shape_x, shape_gamma, dtype and kernel_name
    """
    check_list = ("float16", "float32")
    para_check.check_dtype(params_map.get("dtype"), check_list, param_name="input_dy")

    _check_shape(params_map)


def _check_shape(params_map):
    """
    check parameters including shape_dy, shape_x and shape_gamma
    """
    if operator.ne(tuple(params_map.get("shape_dy")),
                   tuple(params_map.get("shape_x"))):
        error_detail = "shape of input_dy and input_x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_beta_gamma_backprop_v2",
                                                               "input_dy", "input_x", error_detail)

    shape_x = params_map.get("shape_x")
    shape_gamma = params_map.get("shape_gamma")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_gamma, param_name="shape_gamma")

    _check_shape_gamma(shape_x, shape_gamma)


def _check_shape_gamma(shape_x, shape_gamma):
    """
    check if parameter shape_gamma meets the requirements of function
    """
    if len(shape_gamma) > len(shape_x):
        error_detail = "length of shape_gamma can not be longer than shape_x"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_beta_gamma_backprop_v2",
                                                               "input_gamma", "input_x", error_detail)

    for xtem, gamma in zip(reversed(shape_x), reversed(shape_gamma)):
        if xtem != gamma:
            error_detail = "value of shape_gamma is wrong"
            error_manager_vector.raise_err_input_shape_invalid("layer_norm_beta_gamma_backprop_v2",
                                                               "input_gamma", error_detail)


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation
    """
    params_axis_tmp = []
    if len(shape_x) != len(shape_gamma):
        sub = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)
            params_axis_tmp.append(i)

    shape_gamma_new = tuple(shape_gamma)
    params_axis = tuple(params_axis_tmp)

    return shape_gamma_new, params_axis


def layer_norm_beta_gamma_backprop_v2_compute(data_dy, res_for_gamma, output_pd_gamma,
                                              output_pd_beta, shape_gamma,
                                              kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's
    mathematical calculation process

    Parameters
    ----------
    input_dy: TVM tensor
        the placeholder of dy input data
    res_for_gamma: TVM tensor
        the placeholder of x input data
    input_variance: TVM tensor
        the placeholder of variance input data
    input_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma: list or tuple
        original shape of gamma

    Returns
    -------
    res_tuple: tuple
        (pd_gamma, pd_beta)
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(res_for_gamma.shape)
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        res_for_gamma = tbe.cast_to(res_for_gamma, "float32")

    data_x = tbe.vmul(res_for_gamma, data_dy)
    if param_axis:
        pd_gamma, pd_beta = tbe.tuple_sum([data_x, data_dy], param_axis, keepdims=True)
    else:
        pd_beta = tbe.vadds(data_dy, tvm.const(0, dtype=dtype))
        pd_gamma = tbe.vadds(data_x, tvm.const(0, dtype=dtype))

    if dtype == "float16" and not has_improve_precision:
        pd_gamma = tbe.cast_to(pd_gamma, "float32")
        pd_beta = tbe.cast_to(pd_beta, "float32")

    res_list = [pd_gamma, pd_beta]

    return res_list


def update_shape_nz(shape_x):
    """
    function of updating Nz shape

    """
    # ND shape of x >= two dim
    # Nz shape of x >= four dim
    len_x = len(shape_x)
    nz_begin = len_x - 4
    shape_x_nz = []
    for i in range(0, nz_begin):
        shape_x_nz.append(shape_x[i])
    shape_x_nz.append(shape_x[nz_begin])
    shape_x_nz.append(shape_x[nz_begin + 1] * shape_x[nz_begin + 2])
    shape_x_nz.append(shape_x[nz_begin + 2])

    # ND shape of gamma is one dim
    shape_gamma_nz = []
    for i in range(0, nz_begin):
        shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin])
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin + 2])

    param_nz_axis = []
    for i, (xtem, gamma) in enumerate(zip(shape_x_nz, shape_gamma_nz)):
        if xtem != gamma:
            param_nz_axis.append(i)

    param_nz = {"shape_x_nz": shape_x_nz, "shape_gamma_nz": shape_gamma_nz, "param_nz_axis": param_nz_axis}

    return param_nz


def layer_norm_beta_gamma_back_nz_compute(data_dy, data_x, param_nz):
    """
    DSL description of the layer_norm_grad operator's
    mathematical calculation process

    """
    dtype = data_dy.dtype.lower()

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
    data_x = tbe.vmul(data_x, data_dy)
    if param_nz.get("param_nz_axis"):
        pd_gamma, pd_beta = tbe.tuple_sum([data_x, data_dy], param_nz.get("param_nz_axis"), keepdims=True)
    else:
        pd_gamma = tbe.vadds(data_x, tvm.const(0, dtype=dtype))
        pd_beta = tbe.vadds(data_dy, tvm.const(0, dtype=dtype))

    if dtype == "float16" and not has_improve_precision:
        pd_gamma = tbe.cast_to(pd_gamma, "float32")
        pd_beta = tbe.cast_to(pd_beta, "float32")
    res_list = [pd_gamma, pd_beta]

    return res_list


def _update_input_shape(shape_x, shape_dy, shape_gamma):
    """
    update shape_x, shape_dy, shape_gamma
    """
    size_gamma = len(shape_gamma)
    if 1 < size_gamma < len(shape_x):
        shape_x_new = []
        shape_dy_new = []
        shape_gamma_new = []
        sub = len(shape_x) - len(shape_gamma)
        for i in range(sub):
            shape_x_new.append(shape_x[i])
            shape_dy_new.append(shape_dy[i])
        fused_size = functools.reduce(lambda x, y: x * y, shape_gamma)
        shape_x_new.append(fused_size)
        shape_dy_new.append(fused_size)
        shape_gamma_new.append(fused_size)
        return shape_x_new, shape_dy_new, shape_gamma_new

    return shape_x, shape_dy, shape_gamma


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def layer_norm_beta_gamma_backprop_v2(input_dy, res_for_gamma, output_pd_gamma,
                                      output_pd_beta, shape_gamma,
                                      kernel_name="layer_norm_beta_gamma_backprop_v2"):
    """
      algorithm: layernorm_grad
      calculating: gradient of layernorm
                   compute partial derivation of x, gamma and beta
          pd_gamma = np.sum((data_dy*(data_x - data_mean)
                     *np.power((data_variance + EPSLON), (-0.5))),
                     param_axis, keepdims=True)
          pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

      Parameters
      ----------
      input_dy : dict
          shape and dtype of input dy, only support float16, float32
      res_for_gamma: dict
          shape and dtype of input res_for_gamma, only support float16, float32
      output_pd_gamma: dict
          shape and dtype of output, only support float16, float32
      output_pd_beta: dict
          shape and dtype of output, only support float16, float32
      shape_gamma: list
          shape of gamma
      kernel_name: str
          cce kernel name, default value is "layer_norm_beta_gamma_backprop_v2"

      Returns
      -------
      None
      """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = res_for_gamma.get("shape")
    dtype_x = res_for_gamma.get("dtype").lower()

    format_dy = input_dy.get("format")
    attr = {"op_type": "layer_norm_beta_gamma_backprop_v2"}

    if format_dy.upper() == "FRACTAL_NZ":
        param_nz = update_shape_nz(shape_x)

        data_dy = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_dy",
                                  dtype=dtype, attrs=attr)
        data_x = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_x",
                                 dtype=dtype_x)

        res_list = layer_norm_beta_gamma_back_nz_compute(data_dy,
                                                         data_x,
                                                         param_nz)

        tensor_list = [data_dy, data_x, res_list[0], res_list[1]]

        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        tbe.cce_build_code(sch, config)
    else:
        _check_params({"shape_dy": shape_dy, "shape_x": shape_x,
                       "shape_gamma": shape_gamma,
                       "dtype": dtype, "kernel_name": kernel_name})

        shape_x_new, shape_dy_new, shape_gamma_new = _update_input_shape(shape_x, shape_dy, shape_gamma)
        data_dy = tvm.placeholder(shape_dy_new, name="data_dy", dtype=dtype, attrs=attr)
        data_x = tvm.placeholder(shape_x_new, name="data_x", dtype=dtype_x)

        res_list = layer_norm_beta_gamma_backprop_v2_compute(data_dy,
                                                             data_x,
                                                             output_pd_gamma,
                                                             output_pd_beta,
                                                             shape_gamma_new)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res_list)

        tensor_list = [data_dy, data_x] + list(res_list)

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        tbe.cce_build_code(sch, config)
