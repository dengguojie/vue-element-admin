#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("LayerNormGrad", None, None)

case1 = {"params": [{"shape": (3, 8), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (3, 8), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW"},
                    {"shape": (3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "float16", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (3, 8), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (3, 8), "dtype": "int16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (3, 8), "dtype": "int16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (3, 1), "dtype": "int16", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW"},
                    {"shape": (3, 1), "dtype": "int16", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW"},
                    {"shape": (8,), "dtype": "int16", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW"},
                    {"shape": (3, 8), "dtype": "int16", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "int16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 3, 7), "dtype": "int16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 7), "dtype": "int16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "int16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "int16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "int16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "int16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "int16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "int16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 4), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 4),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (1, 3, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 2),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"},
                    {"shape": (1, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 7), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 7),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1),"ori_format": "NCHW"},
                    {"shape": (1, 3, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 1, 1),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"},
                    {"shape": (7,), "dtype": "float16", "format": "NCHW", "ori_shape": (7,),"ori_format": "NCHW"}],
         "case_name": "LayerNormGrad_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)

def calc_expect_func(dy, x, variance, mean, gamma, res_x, res_gamma, res_beta):
    data_dy_ori = dy['value']
    data_x_ori = x['value']
    data_variance_ori = variance['value']
    data_mean_ori = mean['value']
    data_gamma_ori = gamma['value']
    dtype = x['shape']
    dst_type = np.float16 if dtype == 'float16' else np.float32
    shape_dy = dy['shape']
    shape_x = x['shape']
    shape_variance = variance['shape']
    shape_mean = mean['shape']
    shape_gamma = gamma['shape']

    if dtype == "float16":
        data_dy = data_dy_ori.astype(np.float32)
        data_x = data_x_ori.astype(np.float32)
        data_variance = data_variance_ori.astype(np.float32)
        data_mean = data_mean_ori.astype(np.float32)
        data_gamma = data_gamma_ori.astype(np.float32)
    else:
        data_dy = data_dy_ori
        data_x = data_x_ori
        data_variance = data_variance_ori
        data_mean = data_mean_ori
        data_gamma = data_gamma_ori
    param_axis = []
    if len(shape_x) != len(shape_gamma):
        sub_shape = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub_shape):
            shape_gamma.insert(0, 1)
            param_axis.append(i)

    param_axis = tuple(param_axis)

    reduce_axis = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break

    if flag != -1:
        for i in range(flag, len(shape_x)):
            reduce_axis.append(i)
    else:
        reduce_axis.append(len(shape_x) - 1)

    reduce_axis = tuple(reduce_axis)

    m = 1.0
    for i in reduce_axis:
        m *= shape_x[i]

    EPSLON = 1e-12

    pd_xl = data_dy * data_gamma

    pd_var = np.sum(
        ((-0.5) * pd_xl * (data_x - data_mean) * np.power((data_variance + EPSLON), (-1.5))),
        reduce_axis, keepdims=True)

    pd_mean = np.sum(((-1.0) * pd_xl * np.power((data_variance + EPSLON), (-0.5))), reduce_axis,
                     keepdims=True) + \
              pd_var * (1.0 / m) * np.sum(((-2.0) * (data_x - data_mean)), reduce_axis,
                                          keepdims=True)

    pd_x = pd_xl * np.power((data_variance + EPSLON), (-0.5)) + pd_var * (2.0 / m) * (
            data_x - data_mean) + pd_mean * (1.0 / m)
    
    # pd_gamma = np.sum((data_dy * (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))),
    #                   param_axis, keepdims=True)
    # pd_beta = np.sum(data_dy, param_axis, keepdims=True)
    pd_beta = data_dy
    pd_gamma = data_dy * (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))
    
    pd_x_res = pd_x.astype(dst_type)
    pd_gamma_res = pd_gamma.astype(dst_type)
    pd_beta_res = pd_beta.astype(dst_type)

    return pd_x_res, pd_gamma_res, pd_beta_res


precision_case1 = {"params": [{"shape": (3, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW", "param_type": "input", "value_range":[-10,10]},
                    {"shape": (3, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW", "param_type": "input", "value_range":[-10,10]},
                    {"shape": (3, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW", "param_type": "input", "value_range":[0,10]},
                    {"shape": (3, 1), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 1),"ori_format": "NCHW", "param_type": "input", "value_range":[-10,10]},
                    {"shape": (8,), "dtype": "float32", "format": "NCHW", "ori_shape": (8,),"ori_format": "NCHW", "param_type": "input", "value_range":[-10,10]},
                    {"shape": (3, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW", "param_type": "output"},
                    {"shape": (3, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW", "param_type": "output"},
                    {"shape": (3, 8), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 8),"ori_format": "NCHW", "param_type": "output"}],
         "case_name": "LayerNormGrad_3",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case("Ascend910A", precision_case1)

if __name__ == "__main__":
    ut_case.run(["Ascend910A"])
