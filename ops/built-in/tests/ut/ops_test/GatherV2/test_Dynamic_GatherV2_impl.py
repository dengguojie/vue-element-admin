#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

Dynamic GatherV2 ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("GatherV2", "impl.dynamic.gather_v2", "gather_v2")


def test_op_check_supported(test_arg):
    from impl.dynamic.gather_v2 import check_supported_with_reason
    check_supported_with_reason({"shape": [20, 28], "dtype": "int8", "format": "ND", "ori_shape": [20, 28], "ori_format": "ND"},
                     {"shape": [200], "dtype": "int32", "format": "ND", "ori_shape": [200], "ori_format": "ND"},
                     {"shape": [1], "dtype": "int32", "format": "NCHW", "ori_shape": [1], "ori_format": "NCHW"},
                     {"shape": [200, 28], "dtype": "int8", "format": "NCHW", "ori_shape": [200, 28],"ori_format": "ND"})
    check_supported_with_reason({"shape": [20], "dtype": "float16", "format": "ND", "ori_shape": [-2], "ori_format": "ND"},
                     {"shape": [10], "dtype": "int32", "format": "ND", "ori_shape": [10], "ori_format": "ND"},
                     {"shape": [1], "dtype": "int32", "format": "NCHW", "ori_shape": [1], "ori_format": "NCHW"},
                     {"shape": [10], "dtype": "float16", "format": "NCHW", "ori_shape": [10], "ori_format": "ND"})
    check_supported_with_reason({"shape": [30, 5, 61], "dtype": "int32", "format": "ND", "ori_shape": [30, 5, 61],
                     "ori_format": "ND"},
                     {"shape": [10], "dtype": "int32", "format": "ND", "ori_shape": [-2], "ori_format": "ND"},
                     {"shape": [1], "dtype": "int32", "format": "NCHW", "ori_shape": [1], "ori_format": "NCHW"},
                     {"shape": [10, 5, 61], "dtype": "int32", "format": "NCHW", "ori_shape": [10, 5, 61],
                      "ori_format": "ND"})
    check_supported_with_reason({"shape": [10, 1, 28, 16], "dtype": "int64", "format": "ND", "ori_shape": [10, 1, 28, 16],
                     "ori_format": "ND"},
                     {"shape": [10], "dtype": "int32", "format": "ND", "ori_shape": [10], "ori_format": "ND"},
                     {"shape": [1], "dtype": "int32", "format": "NCHW", "ori_shape": [-2], "ori_format": "NCHW"},
                     {"shape": [10, 1, 28, 16], "dtype": "int64", "format": "NCHW", "ori_shape": [10, 1, 28, 16],
                      "ori_format": "ND"})


def gen_dynamic_gather_v2_case(dict_params, dict_indices, dict_axis, dict_y, kernel_name_val, expect):
    return {"params": [dict_params, dict_indices, dict_axis, dict_y],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_v2_case(
                     {"shape": (163623, 80), "dtype": "float32", "ori_shape": (163623, 80),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, 80), "dtype": "float32", "ori_shape": (-1, 80),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (80, 80))},
                     "dynamic_gather_v2_01", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_v2_case(
                     {"shape": (163623, 1), "dtype": "float32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, 1), "dtype": "float32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_v2_02", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_v2_case(
                     {"shape": (163623, 1), "dtype": "float16", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_v2_03", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_v2_case(
                     {"shape": (163623, 1), "dtype": "int32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, 1), "dtype": "int32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_v2_04", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_v2_case(
                     {"shape": (-1, -1), "dtype": "float32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, -1), "dtype": "float32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_v2_05", "success"))

# invalid: y_dtype != params_dtype
ut_case.add_case("all",
                 gen_dynamic_gather_v2_case(
                     {"shape": (163623, 1), "dtype": "float32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (1,), "dtype": "int32", "ori_shape": (1,),
                      "format": "ND", "ori_format": "ND", "range": ((1, 1),)},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_v2_06", RuntimeError))


ut_case.add_cust_test_func(test_func=test_op_check_supported)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
