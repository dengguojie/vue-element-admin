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

Dynamic GatherNd ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("GatherNd", "impl.dynamic.gather_nd", "gather_nd")


def test_ln_import_lib(test_arg):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))

def gen_dynamic_gather_nd_case(dict_params, dict_indices, dict_y, kernel_name_val, expect):
    return {"params": [dict_params, dict_indices, dict_y],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_nd_case(
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (22551, 1), "dtype": "int32", "ori_shape": (22551, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     {"shape": (22551, 1), "dtype": "float32", "ori_shape": (22551, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     "dynamic_gather_nd_01", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_nd_case(
                     {"shape": (22551,), "dtype": "float16", "ori_shape": (22551,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1, 1), "dtype": "int32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((11272, 11272), (1, 1))},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((11272, 11272), (1, 1))},
                     "dynamic_gather_nd_02", "success"))

ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (-1, -1), "dtype": "int32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     {"shape": (-1, 1), "dtype": "int64", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND",  "range": ((22550, 22550), (1, 1))},
                     {"shape": (-1, -1), "dtype": "int32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND",  "range": ((22550, 22550), (1, 1))},
                     "dynamic_gather_nd_03", "success"))

ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (-1, -1), "dtype": "int32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (8, 8))},
                     {"shape": (-1, 1), "dtype": "int64", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND",  "range": ((22550, 22550), (1, 1))},
                     {"shape": (-1, -1), "dtype": "int32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND",  "range": ((22550, 22550), (8, 8))},
                     "dynamic_gather_nd_04", "success"))

# invalid: y_dtype != params_dtype
ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (22551,), "dtype": "float32", "ori_shape": (22551,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1, 1), "dtype": "int32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((11272, 11272), (1, 1))},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((11272, 11272), (1, 1))},
                     "dynamic_gather_nd_05", RuntimeError))

ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (-1, 11, 13, 11, 3), "dtype": "int8", "ori_shape": (8, 11, 13, 11, 3),
                      "format": "ND", "ori_format": "ND", "range": ((8, 8), (11, 11), (13, 13), (11, 11), (3, 3))},
                     {"shape": (5, 7, 16, -1, 3), "dtype": "int32", "ori_shape": (5, 7, 16, 7, 3),
                      "format": "ND", "ori_format": "ND", "range": ((5, 5), (7, 7), (16, 16), (7, 7), (3, 3))},
                     {"shape": (5, 7, 16, -1, 11, 3), "dtype": "int8", "ori_shape": (5, 7, 16, 7, 11, 3),
                      "format": "ND", "ori_format": "ND",
                      "range": ((5, 5), (7, 7), (16, 16), (7, 7), (11, 11), (3, 3))},
                     "dynamic_gather_nd_06", "success"))


ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (-2,), "dtype": "int32", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     {"shape": (-2,), "dtype": "int32", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     {"shape": (-2,), "dtype": "int32", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     "dynamic_gather_nd_03", "success"))

ut_case.add_case("all",
                 gen_dynamic_gather_nd_case(
                     {"shape": (-2,), "dtype": "bool", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     {"shape": (-2,), "dtype": "int32", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     {"shape": (-2,), "dtype": "bool", "ori_shape": (-2,), "format": "ND", "ori_format": "ND"},
                     "dynamic_gather_nd_03", "success"))

ut_case.add_cust_test_func(test_func=test_ln_import_lib)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
