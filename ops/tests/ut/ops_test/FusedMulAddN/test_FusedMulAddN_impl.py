#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("FusedMulAddN", None, None)

def calc_expect_func(x, y, z, output):
    res = x["value"] * z["value"] + y["value"]
    res = res.astype(output['dtype'])
    return res

case1 = {"params": [
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)},
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)},
    {"shape": (), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": ()},
    {"shape": (1, 4, 1, 1, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4, 1, 1, 16)}],
    "case_name": "fused_mul_add_n_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

case2 = {"params": [
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)},
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)},
    {"shape": (), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": ()},
    {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16)}],
    "case_name": "fused_mul_add_n_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

precision_case1 = {"params": [{"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16), "param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"output"}],
                              "expect": "success",
                              "calc_expect_func": calc_expect_func,
                              "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

ut_case.add_precision_case("Ascend910", precision_case1)


if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")

