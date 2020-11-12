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

FusedMulAddN ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

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

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

precision_case1 = {"params": [{"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case2 = {"params": [{"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ), "param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (33, ), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (33, ),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case3 = {"params": [{"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32), "param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (16, 16, 64, 32), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (16, 16, 64, 32),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

precision_case4 = {"params": [{"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16), "param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"input"},
                              {"shape": (1,), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,),"param_type":"input"},
                              {"shape": (4, 4, 16, 16), "dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4, 4, 16, 16),"param_type":"output"}],
                   "expect": "success",
                   "calc_expect_func": calc_expect_func,
                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case1)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case2)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case3)
ut_case.add_precision_case(["Ascend310", "Ascend910"], precision_case4)

