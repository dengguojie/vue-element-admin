#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("FusedMulApplyMomentum", None, None)

def calc_expect_func(var, accum, lr, x1, momentum, x2, output_var, output_accum):
    grad = x2["value"] * x1["value"]
    accum_delta = accum["value"] * momentum["value"]
    accum_t = grad + accum_delta

    var_delta = accum_t * lr["value"]
    var_t = var["value"] - var_delta
    var_t = var_t.astype(output_var["dtype"])
    accum_t = accum_t.astype(output_accum["dtype"])

    return var_t, accum_t

case1 = {"params": [{"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float32", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float32","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)}],
         "case_name": "fused_mul_apply_momentum_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                    {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)}],
         "case_name": "fused_mul_apply_momentum_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

precision_case1 = {"params": [{"shape": (144,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (144,16,16,16), "param_type":"input"},
                              {"shape": (144,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (144,16,16,16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (144,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (144,16,16,16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (144,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (144,16,16,16),"param_type":"output"},
                              {"shape": (144,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (144,16,16,16),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case("Ascend910", precision_case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
