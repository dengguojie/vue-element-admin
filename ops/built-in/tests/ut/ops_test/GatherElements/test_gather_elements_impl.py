#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

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


ut_case = OpUT("GatherElements", "impl.dynamic.gather_elements", "gather_elements")

from impl.dynamic.gather_elements import check_supported

def test_check_support(test_arg):
    print(test_arg)
    res, reason = check_supported({"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 1024), "shape": (1024, 1024),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 1024), "shape": (1024, 1024),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1024, 1024), "shape": (1024, 1024),
              "param_type": "output"},
            -1)
    # assert not res

    res, reason = check_supported({"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (25600,), "shape": (25600,),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (10,), "shape": (10,),
              "param_type": "input"},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (10,), "shape": (10,),
              "param_type": "output"},
            0)
    # assert res

    res, reason = check_supported({"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 101, 2), "shape": (2, 101, 2),
              "param_type": "input"},
                {"dtype": "int64", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2, 2), "shape": (2, 2, 2),
              "param_type": "input"},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2, 2, 2), "shape": (2, 2, 2),
              "param_type": "output"},
            1)
    # assert res

    res, reason = check_supported({"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (64, 64), "shape": (64, 64),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (10000, 64), "shape": (10000, 64),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (10000, 64), "shape": (10000, 64),
              "param_type": "output"},
            0)
    # assert res

    res, reason = check_supported({"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (25, 25, 25, 3), "shape": (25, 25, 25, 3),
              "param_type": "input"},
                {"dtype": "int64", "format": "ND", "ori_format": "ND", "ori_shape": (25, 25, 25, 3), "shape": (25, 25, 25, 3),
              "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (25, 25, 25, 3), "shape": (25, 25, 25, 3),
              "param_type": "output"},
            -1)
    # assert not res

ut_case.add_cust_test_func(test_func=test_check_support)

ut_case.add_case("all", {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, ), "shape": (-1, ),
              "param_type": "input", "range": ((22551, 22551),)},
                {"dtype": "int64", "format": "ND", "ori_format": "ND", "ori_shape": (-1, ), "shape": (-1, ),
              "param_type": "input", "range": ((22551, 22551),)},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, ), "shape": (-1, ),
              "param_type": "output", "range": ((22551, 22551),)},
            0],
    "case_name": "dynamic_01"
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "input", "range": ((22551, 22551), (1, 1))},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "input", "range": ((22551, 22551), (1, 1))},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "output", "range": ((22551, 22551), (1, 1))},
            0],
    "case_name": "dynamic_02"
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "input", "range": ((1024, 1024), (1024, 1024))},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "input", "range": ((1024, 1024), (1024, 1024))},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1), "shape": (-1, -1),
              "param_type": "output", "range": ((1024, 1024), (1024, 1024))},
            0],
    "case_name": "dynamic_03"
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "input", "range": ((32, 32), (32, 32), (32, 32))},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "input", "range": ((32, 32), (32, 32), (32, 32))},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "output", "range": ((32, 32), (32, 32), (32, 32))},
            0],
    "case_name": "dynamic_04"
})

ut_case.add_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "input", "range": ((25, 25), (32, 32), (39, 39))},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "input", "range": ((25, 25), (32, 32), (39, 39))},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (-1, -1, -1), "shape": (-1, -1, -1),
              "param_type": "output", "range": ((25, 25), (32, 32), (39, 39))},
            0],
    "case_name": "dynamic_05"
})


if __name__ == "__main__":
    ut_case.run("Ascend310")
    ut_case.run("Ascend910A")