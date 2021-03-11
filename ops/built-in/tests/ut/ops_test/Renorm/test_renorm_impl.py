# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import torch
import numpy as np
ut_case = BroadcastOpUT("renorm")

#pylint: disable=unused-argument
def calc_expect_func(input_x, output_z, p, dim, maxnorm):
    dtype = input_x["dtype"]
    shape = input_x["shape"]
    dims = len(shape)
    input_x_tmp = input_x["value"]
    if dtype == "float16":
        input_x_tmp.astype(np.float32)
    ext = 1e-7
    shape_list = []
    for i in range(dims):
        if i != dim:
            shape_list = shape_list + [i]
    print("shape_list = ", shape_list)
    if p == 1:
        x_sum = np.sum(np.absolute(input_x_tmp), tuple(shape_list), keepdims=True)
        x_l1norm = np.minimum(x_sum, maxnorm)
        ratio = np.divide(x_l1norm, np.add(x_sum, ext))
    elif p == 2:
        x_square = np.multiply(input_x_tmp, input_x_tmp)
        x_square_sum = np.sum(x_square, tuple(shape_list), keepdims=True)

        x_l2norm_sqrt = np.sqrt(x_square_sum)
        x_l2norm = np.minimum(x_l2norm_sqrt, maxnorm)

        ratio = np.divide(x_l2norm, np.add(x_l2norm_sqrt, ext))
    else:
        if p == 0:
            x_tmp = np.sum(input_x_tmp, tuple(shape_list), keepdims=True)
            x_tmp.astype(np.float32)
        else:
            p_log = np.log(np.absolute(input_x_tmp))
            p_mul = np.multiply(p_log, p)
            x_sum = np.exp(p_mul)
            x_psum = np.sum(x_sum, tuple(shape_list), keepdims=True)
            p_log_v = np.log(x_psum)
            p_mul_v = np.multiply(p_log_v, 1 / p)
            x_tmp = np.exp(p_mul_v)
        x_lpnorm_p = np.minimum(x_tmp, maxnorm)
        x_tmp_ext = np.add(x_tmp, ext)
        ratio = np.divide(x_lpnorm_p, x_tmp_ext)
    if input_x == "float16":
        ratio.astype(np.float16)
    return ratio

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 3), "shape": (3, 3, 3),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1, 1), "shape": (3, 1, 1),
                "param_type": "output"}, 2.0, 0, 3.0],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3), "shape": (3, 3),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1), "shape": (3, 1),
                "param_type": "output"}, 2.0, 0, 3.0],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 3), "shape": (3, 3, 3),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1, 1), "shape": (3, 1, 1),
                "param_type": "output"}, 1.0, 0, 3.0],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 3), "shape": (3, 3, 3),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1, 1), "shape": (3, 1, 1),
                "param_type": "output"}, 3.0, 0, 3.0],
    "calc_expect_func": calc_expect_func
})