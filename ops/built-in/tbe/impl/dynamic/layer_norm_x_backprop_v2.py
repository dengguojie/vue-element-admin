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
layer_norm_x_backprop_v2
"""
# pylint: disable=too-many-lines
import operator

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import operation


# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648

# Minimum positive number greater than 0
EPSLON = 1e-12

import tbe as mytbe
@mytbe.common.register.register_param_generalization("LayerNormXBackpropV2")
def layer_norm_x_backprop_generalization(input_dy, input_x, input_variance,
                                         input_mean, input_gamma, output_pd_x, output_res_gamma,
                                         impl_mode, generalize_config={"mode": "keep_rank"}):
    # for now only support dy and x is (-1, -1, N), variavce and mean is (-1, -1, 1),
    # shape_gamma is (N,)
    result = []
    x_shape = input_x["shape"]
    last_dim = x_shape[-1]
    input_dy["shape"] = [-1, -1, last_dim]
    input_x["shape"] = [-1, -1, last_dim]
    input_variance["shape"] = [-1, -1, 1]
    input_mean["shape"] = [-1, -1, 1]
    input_gamma["shape"] = [last_dim]
    result.append([input_dy, input_x, input_variance, input_mean, input_gamma, output_pd_x, output_res_gamma])
    return result


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    shape_gamma_new: tuple
        new shape_gamma after update
    params_axis: tuple
        the list of axis for gamma reduce_sum
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


def _get_data_gm(shapes, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma

    Parameters
    ----------
    shapes: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma}
    dtype: str
        the data type

    Returns
    -------
    data_gm: tuple
        (data_dy, data_x, data_variance, data_mean, data_gamma)
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"), name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"), name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_params(shape_x, shape_mean, shape_gamma):
    """
    compute parameters including param_axis, reduce_axis and mean_num

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    """
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

    reduce_axis_tmp = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break
    if flag != -1:
        for i in range(flag, len(shape_x)):
            reduce_axis_tmp.append(i)
    else:
        reduce_axis_tmp.append(len(shape_x) - 1)
    reduce_axis = tuple(reduce_axis_tmp)

    mean_num = 1.0
    for i in reduce_axis:
        mean_num *= shape_x[i]

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis, "mean_num": mean_num}

    return params


def _get_pd_xl(data, shape_x):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_xl: tvm.tensor
        data_dy*data_gamma
    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front(data, cast_dtype):
    """
    compute front part of pd_var according to data_variance

    Parameters
    ----------
    data: dict
        placeholders after cast

    Returns
    -------
    pd_var_1: tvm.tensor
        np.power((data_variance + EPSLON), (-1.5))
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    """
    var_elta = tbe.vadds(data.get("data_variance"), tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log, tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var(data, params, shape_x, pd_xl, cast_dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    pd_xl: tvm.tensor
        data_dy*data_gamma

    Returns
    -------
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    pd_var_1, var_elta_2 = _get_pd_var_front(data, cast_dtype)
    data_mean_cast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(pd_xl, sub_x_mean)
    pdvar_sum = tbe.reduce_sum(pdvar_mul1, params.get("reduce_axis"), keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)
    res_for_gamma = tbe.vmul(sub_x_mean, var_elta_2_cast)

    return pd_var, var_elta_2, sub_x_mean, res_for_gamma


def _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var, var_elta_2
    and sub_x_mean

    Parameters
    ----------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    pd_xl: tvm.tensor
        data_dy*data_gamma
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean

    Returns
    -------
    pd_mean: tvm.tensor
        np.sum(((-1.0)*pd_xl
        *np.power((data_variance + EPSLON), (-0.5))), reduce_axis,
        keepdims=True)
        + pd_var*(1.0/m)*np.sum(((-2.0)*(data_x - data_mean)),
        reduce_axis, keepdims=True)
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    pdmean1_sum = tbe.reduce_sum(pd_xl, params.get("reduce_axis"), keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean = tbe.vmuls(pdmean1_mul, tvm.const(-1.0, dtype=cast_dtype))
    return pd_mean


def _get_pd_x_front(data, params, shape_x, cast_dtype):
    """
    compute front part of pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_x_1: tvm.tensor
        pd_xl*np.power((data_variance + EPSLON), (-0.5))
    pd_x_2: tvm.tensor
        pd_var*(2.0/m)*(data_x - data_mean)
    pd_x_3: tvm.tensor
        pd_mean*(1.0/m)
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    pd_xl = _get_pd_xl(data, shape_x)
    pd_var, var_elta_2, sub_x_mean, res_for_gamma = _get_pd_var(data, params, shape_x, pd_xl, cast_dtype)
    pd_mean = _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype)
    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    pdx2_broad = tbe.broadcast(pd_var, shape_x)
    pdx2_mul = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_2 = tbe.vmuls(pdx2_mul, tvm.const((2*(params.get("mean_num")**(-1))), dtype=cast_dtype))
    pd_x_3 = tbe.vmuls(pd_mean, tvm.const((params.get("mean_num")**(-1)), dtype=cast_dtype))

    return pd_x_1, pd_x_2, pd_x_3, res_for_gamma


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    var_elta_2_cast: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    pd_x_1, pd_x_2, pd_x_3, res_for_gamma = _get_pd_x_front(data, params, shape_x, cast_dtype)

    pdx_broad = tbe.broadcast(pd_x_3, shape_x)
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x_ub = tbe.vadd(pdx_add, pdx_broad)

    if dtype == "float16" and cast_dtype == "float32":
        pd_x = tbe.cast_to(pd_x_ub, dtype)
    else:
        return pd_x_ub, res_for_gamma

    return pd_x, res_for_gamma


def _get_res(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    pd_x, res_for_gamma = _get_pd_x(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _get_pds(data_dy, data_x, data_variance, data_mean,
             data_gamma, shape_gamma_ori):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_variance: TVM tensor
        the placeholder of variance input data
    data_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma_ori: list or tuple
        original shape of gamma

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    res_for_gamma: tvm.tensor
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)
    shape_mean = shape_util.shape_to_list(data_mean.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        cast_dtype = "float32"

    params = _get_params(shape_x, shape_mean, shape_gamma_ori)

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {"data_dy": data_dy, "data_x": data_x,
            "data_variance": data_variance,
            "data_mean": data_mean, "data_gamma": data_gamma}

    pd_x, res_for_gamma = _get_res(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def layer_norm_x_backprop_v2_compute(input_dy, input_x,
                                     input_variance, input_mean,
                                     input_gamma, output_pd_x, output_res_gamma,
                                     kernel_name="layer_norm_x_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's mathematical
    calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    res_tuple: tuple
        (pd_x, res_for_gamma)
    """
    pd_x, res_for_gamma = _get_pds(input_dy, input_x, input_variance, input_mean, input_gamma, input_gamma.shape)
    res_list = [pd_x, res_for_gamma]

    return res_list


@register_operator("LayerNormXBackpropV2",pattern = "Layer_norm_x_backprop_v2")
def layer_norm_x_backprop_v2(input_dy, input_x, input_variance, input_mean,
                             input_gamma, output_pd_x, output_res_gamma,
                             kernel_name="layer_norm_x_backprop_v2"):
    """
    algorithm: layernorm_x_backprop_v2
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        pd_xl    = data_dy*data_gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((data_variance + EPSLON), (-0.5))),
                   reduce_axis, keepdims=True)
                   + pd_var*(1.0/m)
                   *np.sum(((-2.0)*(data_x - data_mean)),
                   reduce_axis, keepdims=True)
        pd_x     = pd_xl*np.power((data_variance + EPSLON), (-0.5))
                   + pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
        res_for_gamma = 
        (data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    last_dim = shape_x[-1]
    shape_variance = input_variance.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")

    dynamic_shape_dy = shape_dy
    dynamic_shape_variance = shape_variance
    dynamic_shape_gamma = shape_gamma
    dim_0 = operation.var("dim_0")
    dim_1 = operation.var("dim_1")
    dynamic_shape_dy = (dim_0, dim_1, last_dim)
    dynamic_shape_x = dynamic_shape_dy
    dynamic_shape_variance = (dim_0, dim_1, 1)
    dynamic_shape_mean = dynamic_shape_variance
    with tbe.compute():
        dynamic_shape_gamma = (1, 1, last_dim)
        data_gm = _get_data_gm({"shape_dy": dynamic_shape_dy, "shape_x": dynamic_shape_x,
                                "shape_var": dynamic_shape_variance,
                                "shape_mean": dynamic_shape_mean,
                                "shape_gamma": dynamic_shape_gamma}, dtype)

        res_list = layer_norm_x_backprop_v2_compute(data_gm[0], data_gm[1],
                                                    data_gm[2], data_gm[3],
                                                    data_gm[4], output_pd_x,
                                                    output_res_gamma)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res_list)

    tensor_list = list(data_gm) + list(res_list)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.build(sch, config)
