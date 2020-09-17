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

ScatterNonAliasingAdd ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterNonAliasingAdd", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============

def gen_scatter_nd_add_case(x_shape, indices_shape, updates_shape,
                            dtype_x, case_name_val, expect):
    return {
        "params":
            [
                {
                    "shape": x_shape,
                    "dtype": dtype_x,
                    "ori_shape": x_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": indices_shape,
                    "dtype": "int32",
                    "ori_shape": indices_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": updates_shape,
                    "dtype": dtype_x,
                    "ori_shape": updates_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": x_shape,
                    "dtype": dtype_x,
                    "ori_shape": x_shape,
                    "ori_format": "ND",
                    "format": "ND"
                }
            ],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True
    }


ut_case.add_case("all",
                 gen_scatter_nd_add_case((33, 5), (33, 25, 1), (33, 25, 5),
                                         "float32", "valid_fp32", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 14, 16), (10, 2, 1), (10, 2, 14, 16),
                                         "float16", "valid_fp16", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((8, 427), (2, 3, 1), (2, 3, 427),
                                         "int32", "valid_int32", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 512, 7, 7), (128, 1), (128, 512, 7, 7),
                                         "float16", "valid_fp16_2", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 32), (128, 1), (128, 32),
                                         "float32", "valid_fp32_2", "success"))

# ============ auto gen ["Ascend910"] test cases end =================

if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
