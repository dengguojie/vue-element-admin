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

FusedMulApplyMomentum ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

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


def test_get_op_support_info(test_arg):
    from impl.fused_mul_apply_momentum import get_op_support_info
    get_op_support_info({"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)})

ut_case.add_case(["Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend710", "Ascend910"], case2)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (3,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,16,16,16), "param_type":"input"},
               {"shape": (3,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,16,16,16),"param_type":"input"},
               {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
               {"shape": (3,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,16,16,16),"param_type":"input"},
               {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
               {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
               {"shape": (3,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,16,16,16),"param_type":"output"},
               {"shape": (3,16,16,16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3,16,16,16),"param_type":"output"}],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"output"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"output"}],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ), "param_type":"input"},
               {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"output"},
               {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"output"}],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (33, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 32), "param_type":"input"},
               {"shape": (33, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 32),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (33, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 32),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
               {"shape": (33, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 32),"param_type":"output"},
               {"shape": (33, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, 32),"param_type":"output"}],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)})

