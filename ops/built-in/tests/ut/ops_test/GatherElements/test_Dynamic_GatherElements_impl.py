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

Dynamic GatherElements ut case
"""
from op_test_frame.ut import OpUT


ut_case = OpUT("GatherElements", "impl.dynamic.gather_elements", "gather_elements")


def gen_dynamic_gather_elements_case(dict_params, dict_indices, dict_y, axis, kernel_name_val, expect):
    return {"params": [dict_params, dict_indices, dict_y, axis],
            "case_name": kernel_name_val,
            "expect": expect,
            "support_expect": True}


ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_elements_case(
                     {"shape": (163623, 80), "dtype": "float32", "ori_shape": (163623, 80),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (80, 80))},
                     {"shape": (-1,), "dtype": "int32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((80, 80),)},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((80, 80),)},
                      0,
                     "dynamic_gather_elements_01", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_elements_case(
                     {"shape": (163623, 1), "dtype": "float32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                      0,
                     "dynamic_gather_elements_02", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_elements_case(
                     {"shape": (163623, 1), "dtype": "float16", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                      0,
                     "dynamic_gather_elements_03", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_elements_case(
                     {"shape": (163623, 1), "dtype": "int32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1, 1), "dtype": "int32", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                      1,
                     "dynamic_gather_elements_04", "success"))

ut_case.add_case(["Ascend910A"],
                 gen_dynamic_gather_elements_case(
                     {"shape": (-1, -1), "dtype": "float32", "ori_shape": (-1, -1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},                      
                     {"shape": (-1,), "dtype": "float32", "ori_shape": (-1,),
                     "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                     0,
                     "dynamic_gather_elements_05", "success"))

# invalid: y_dtype != params_dtype
ut_case.add_case("all",
                 gen_dynamic_gather_elements_case(
                     {"shape": (163623, 1), "dtype": "float32", "ori_shape": (163623, 1),
                      "format": "ND", "ori_format": "ND", "range": ((163623, 163623), (1, 1))},
                     {"shape": (-1,), "dtype": "int64", "ori_shape": (-1,),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551),)},
                     {"shape": (-1, 1), "dtype": "float16", "ori_shape": (-1, 1),
                      "format": "ND", "ori_format": "ND", "range": ((22551, 22551), (1, 1))},
                      0,
                     "dynamic_gather_elements_06", RuntimeError))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
