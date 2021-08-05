# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from te import platform as tbe_platform
from te import tvm
import te.lang.cce as tbe
from te.utils import shape_util

EPSLON = 1e-12
warnings.filterwarnings("ignore")


def _update_gamma_shape(shape_x, shape_gamma):
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


def _get_params(shape_x, shape_mean, shape_gamma):
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

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis,
              "mean_num": mean_num}

    return params


def _get_pd_var_front(data, cast_dtype):
    var_elta = tbe.vadds(data.get("data_variance"),
                         tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log,
                             tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var(data, params, shape_x, pd_xl, cast_dtype):
    pd_var_1, var_elta_2 = _get_pd_var_front(data, cast_dtype)

    data_mean_cast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(sub_x_mean, pd_xl)
    pdvar_sum = tbe.sum(pdvar_mul1, params.get("reduce_axis"),
                        keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    return pd_var, var_elta_2, sub_x_mean


def _get_pd_xl(data, shape_x):
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))
    return pd_xl


def _get_pd_mean(params, pd_xl, var_elta_2, cast_dtype):

    pdmean1_sum = tbe.sum(pd_xl, params.get("reduce_axis"),
                          keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = tbe.vmuls(pdmean1_mul,
                          tvm.const(-1.0, dtype=cast_dtype))
    return pd_mean_1


def _get_pd_x_front(data, params, shape_x, cast_dtype):
    pd_xl = _get_pd_xl(data, shape_x)

    pd_var, var_elta_2, sub_x_mean = _get_pd_var(data, params, shape_x, pd_xl,
                                                 cast_dtype)

    pd_mean = _get_pd_mean(params, pd_xl, var_elta_2, cast_dtype)
    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    res_for_gamma = tbe.vmul(var_elta_2_cast, sub_x_mean)

    pd_var = tbe.vmuls(pd_var, tvm.const((2 * (params.get("mean_num") ** (-1))), dtype=cast_dtype))
    pdx2_broad = tbe.broadcast(pd_var, shape_x)
    pd_x_2 = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_3 = tbe.vmuls(pd_mean, tvm.const((params.get("mean_num") ** (-1)), dtype=cast_dtype))

    return pd_x_1, pd_x_2, pd_x_3, res_for_gamma


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
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
    pd_x, res_for_gamma = _get_pd_x(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _get_pds(data_dy, data_x, data_variance, data_mean,
             data_gamma, shape_gamma_ori):
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)
    shape_mean = shape_util.shape_to_list(data_mean.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.vexp", "float32"):
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


def _get_data_gm(shapes, dtype):
    data_dy = tvm.placeholder(shapes.get("shape_dy"),
                              name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"),
                             name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"),
                                    name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"),
                                name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"),
                                 name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def layer_norm_x_backprop_v2_compute(input_dy, input_x,
                                     input_variance, input_mean,
                                     input_gamma):
    pd_x, res_for_gamma = _get_pds(input_dy, input_x, input_variance, input_mean,
                                   input_gamma, input_gamma.shape)
    res_list = [pd_x, res_for_gamma]

    return res_list


def layer_norm_x_backprop_v2(input_dy, input_x, input_variance, input_mean,
                             input_gamma, output_pd_x,res_for_gamma,
                             kernel_name="layer_norm_x_backprop_v2"):
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_variance = input_variance.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")

    format_dy = input_dy.get("format")

    shape_gamma = _update_gamma_shape(shape_x, shape_gamma)[0]

    data_gm = _get_data_gm({"shape_dy": shape_dy, "shape_x": shape_x,
                            "shape_var": shape_variance,
                            "shape_mean": shape_mean,
                            "shape_gamma": shape_gamma}, dtype)

    res_list = layer_norm_x_backprop_v2_compute(data_gm[0], data_gm[1],
                                                data_gm[2], data_gm[3],
                                                data_gm[4])

    with tvm.target.cce():
        sch = tbe.auto_schedule(res_list)

    tensor_list = list(data_gm) + list(res_list)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}

    tbe.cce_build_code(sch, config)


ut_case = OpUT("layer_norm_grad", "layer_norm_grad.test_static_layer_norm_grad_impl",
               "layer_norm_x_backprop_v2")

case1 = {
    "params": [{"shape": (1024,8192), "dtype": "float16", "format": "ND", "ori_shape":(1024,8192),"ori_format":"ND"},
               {"shape": (1024,8192), "dtype": "float16", "format": "ND", "ori_shape":(1024,8192),"ori_format":"ND"},
               {"shape": (1024,1), "dtype": "float16", "format": "ND", "ori_shape":(1024,1),"ori_format":"ND"},
               {"shape": (1024,1), "dtype": "float16", "format": "ND", "ori_shape":(1024,1),"ori_format":"ND"},
               {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape":(8192,),"ori_format":"ND"},
               {"shape": (1024,8192), "dtype": "float16", "format": "ND", "ori_shape":(1024,8192),"ori_format":"ND"},
               {"shape": (1024,8192), "dtype": "float16", "format": "ND", "ori_shape":(1024,8192),"ori_format":"ND"}
               ],
    "case_name": "test_layer_norm_grad_1",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
]

for item in compile_case:
    ut_case.add_case(["Ascend910A"], case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
