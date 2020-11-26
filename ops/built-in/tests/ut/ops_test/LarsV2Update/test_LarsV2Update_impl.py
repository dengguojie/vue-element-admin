"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

LarsV2Update ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("LarsV2Update", None, None)

case1 = {"params": [{"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
                    {"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)}],
         "case_name": "lars_v2_update_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 1, 512, 128), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
                    {"shape": (1, 1, 512, 128), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1, 1, 512, 128), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1, 1, 512, 128)}],
         "case_name": "lars_v2_update_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)


def calc_expect_func(weight, grad, weight_s, grad_s, weight_decay, learning_rate, out, hyperparam, epsilon, use_clip):
    dtype = weight["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    weight = weight["value"]
    grad = grad["value"]
    weight_s = weight_s["value"]
    grad_s = grad_s["value"]
    weight_decay = weight_decay["value"]
    lr = learning_rate["value"]

    if sdtype == np.float16:
        weight = weight.astype(np.float32)
        grad = grad.astype(np.float32)
        weight_s = weight_s.astype(np.float32)
        grad_s = grad_s.astype(np.float32)
        weight_decay = weight_decay.astype(np.float32)
        lr = lr.astype(np.float32)

    weight_norm = np.sqrt(weight_s)
    grad_norm = np.sqrt(grad_s)

    hyper_weight_norm = hyperparam * weight_norm
    g_norm_wd = grad_norm + weight_decay * weight_norm + epsilon
    g_wd = grad + weight_decay * weight
    if use_clip:
        coeff = hyper_weight_norm / g_norm_wd
        coeff = coeff / lr
        coeff_min = np.maximum(coeff, 0)
        clip_coeff = np.minimum(coeff_min, 1)
    else:
        clip_coeff = hyper_weight_norm / g_norm_wd
    out = clip_coeff * g_wd
    if sdtype == np.float16:
        out = out.astype(sdtype)
    return [out]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 512, 128), "shape": (1, 1, 512, 128), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 512, 128), "shape": (1, 1, 512, 128), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 512, 128), "shape": (1, 1, 512, 128), "param_type": "output"},
               0.001, 1e-5, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 3, 64), "shape": (7, 7, 3, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 3, 64), "shape": (7, 7, 3, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 3, 64), "shape": (7, 7, 3, 64), "param_type": "output"},
               0.001, 1e-5, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 64, 64), "shape": (3, 3, 64, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 64, 64), "shape": (3, 3, 64, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 3, 64, 64), "shape": (3, 3, 64, 64), "param_type": "output"},
               0.001, 1e-5, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 128, 64), "shape": (7, 7, 128, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 128, 64), "shape": (7, 7, 128, 64), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (7, 7, 128, 64), "shape": (7, 7, 128, 64), "param_type": "output"},
               0.001, 1e-5, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
