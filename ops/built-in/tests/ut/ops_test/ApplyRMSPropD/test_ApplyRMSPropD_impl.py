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

ApplyRmsPropD ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("ApplyRmsPropD", None, None)

case1 = {"params": [{"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128,16), "dtype": "float32", "format": "ND", "ori_shape": (128,16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float32", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128,16), "dtype": "float16", "format": "ND", "ori_shape": (128,16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    {"shape": (128, 16), "dtype": "float16", "format": "ND", "ori_shape": (128, 16),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    {"shape": (128,), "dtype": "float32", "format": "ND", "ori_shape": (128,),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    {"shape": (811, 12, 73, 5), "dtype": "float32", "format": "ND", "ori_shape": (811, 12, 73, 5),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    {"shape": (1, 1, 16, 96), "dtype": "float32", "format": "ND", "ori_shape": (1, 1, 16, 96),"ori_format": "ND"},
                    0.9, 0.9, 1.0e-7],
         "case_name": "apply_rms_prop_d_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)


def calc_expect_func(var,
                     ms,
                     mom,
                     lr,
                     grad,
                     out_var,
                     out_ms,
                     out_mom,
                     rho,
                     momentum,
                     epsilon):
    dtype = var["dtype"]
    if dtype == "fp16" or dtype == "float16":
        sdtype = np.float16
    elif dtype == "fp32" or dtype == "float32":
        sdtype = np.float32
    else:
        raise RuntimeError("unsupported dtype:%s " % dtype)

    var_in_data = var["value"]
    ms_in_data = ms["value"]
    mom_in_data = mom["value"]
    lr_in_data = lr["value"]
    grad_in_data = grad["value"]

    ms_out_data = rho * ms_in_data + (1 - rho) * grad_in_data * grad_in_data
    mom_out_data = momentum * mom_in_data + lr_in_data * grad_in_data / np.sqrt(ms_out_data + epsilon)
    var_out_data = var_in_data - mom_out_data

    return [var_out_data.astype(sdtype), ms_out_data.astype(sdtype), mom_out_data.astype(sdtype)]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "output"},
               0.9, 0.9, 1.0e-7],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (128, ), "shape": (128, ), "param_type": "output"},
               0.9, 0.9, 1.0e-7],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 16, 96), "shape": (1, 1, 16, 96), "param_type": "output"},
               0.9, 0.9, 1.0e-7],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "output"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (11, 12, 73, 5), "shape": (11, 12, 73, 5), "param_type": "output"},
               0.9, 0.9, 1.0e-7],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run("Ascend910")


