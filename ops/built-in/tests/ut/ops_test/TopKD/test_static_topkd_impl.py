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

Dynamic Topk ut case
"""
import te
from op_test_frame.ut import OpUT
from unittest.mock import MagicMock
from unittest.mock import patch
from impl.top_k_d import top_k_d
from tbe.common.platform.platform_info import set_current_compile_soc_info


ut_case = OpUT("TopkD", "impl.top_k_d", "top_k_d")

case1 = {"params": [{"shape": (8, 32), "dtype": "float16", "ori_shape": (8, 32), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8,8), "dtype": "float16", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8,8), "dtype": "int32", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    8,True,-1,True],
         "case_name": "TopkD_k_8",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (8, 32), "dtype": "float16", "ori_shape": (8, 32), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8,8), "dtype": "float16", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8,8), "dtype": "int32", "format": "ND", "ori_shape": (8,8),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    8,True,-1,False],
         "case_name": "TopkD_k_8",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (8, 5000), "dtype": "float16", "ori_shape": (8, 5000), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (8192,), "dtype": "float16", "format": "ND", "ori_shape": (8192,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8,6), "dtype": "float16", "format": "ND", "ori_shape": (8,6),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8,6), "dtype": "int32", "format": "ND", "ori_shape": (8,6),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    6,True,-1,True],
         "case_name": "TopkD_k_6",
         "expect": "RuntimeError",
         "support_expect": False}

case4 = {"params": [{"shape": (8, 6000), "dtype": "float16", "ori_shape": (8, 6000), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (2048,), "dtype": "float16", "format": "ND", "ori_shape": (2048,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (8, 5000), "dtype": "float16", "format": "ND", "ori_shape": (8, 5000),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (8, 5000), "dtype": "int32", "format": "ND", "ori_shape": (8, 5000),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    5000,True,-1,False],
         "case_name": "TopkD_k_8",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (1, 6000), "dtype": "float16", "ori_shape": (1, 32), "format": "ND", "ori_format": "ND", "range": ((1, 880), (1, 48))},
                    {"shape": (2048,), "dtype": "float16", "format": "ND", "ori_shape": (2048,),"ori_format": "ND", "range": ((8192,8192),)},
                    {"shape": (1,5000), "dtype": "float16", "format": "ND", "ori_shape": (1,5000),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    {"shape": (1,5000), "dtype": "int32", "format": "ND", "ori_shape": (1,5000),"ori_format": "ND", "range": ((1, 880), (1, 16))},
                    5000,True,-1,False],
         "case_name": "TopkD_k_8",
         "expect": "success",
         "support_expect": True}
# TODO fix me, this comment, run failed
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend310P3"], case1)
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend310P3"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend310P3"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend310P3"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend610", "Ascend310P3"], case5)

def test_static_ascend310p3(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.top_k_d import top_k_d
    te_set_version("Ascend610", "VectorCore")
    top_k_d({"shape": (8, 32), "format": "ND", "dtype": "float16", "ori_shape": (8, 32), "ori_format": "ND"},
          {"shape": (8192, ), "format": "ND", "dtype": "float16", "ori_shape": (8192, ), "ori_format": "ND"},
          {"shape": (8, 8), "format": "ND", "dtype": "float16", "ori_shape": (8, 8), "ori_format": "ND"},
          {"shape": (8, 8), "format": "ND", "dtype": "int32", "ori_shape": (8, 8), "ori_format": "ND"},
          8, True, -1, True)
    te_set_version(test_arg)




def test_static_lhisi(test_arg):
    set_current_compile_soc_info("Hi3796CV300CS")
    top_k_d({"shape": (1000000, ), "format": "ND", "dtype": "float16", "ori_shape": (1000000, ), "ori_format": "ND"},
            {"shape": (8192, ), "format": "ND", "dtype": "float16", "ori_shape": (8192, ), "ori_format": "ND"},
            {"shape": (100, ), "format": "ND", "dtype": "float16", "ori_shape": (100, ), "ori_format": "ND"},
            {"shape": (100, ), "format": "ND", "dtype": "int32", "ori_shape": (100, ), "ori_format": "ND"},
            100, True, -1, True)
    set_current_compile_soc_info(test_arg)

ut_case.add_cust_test_func(test_func=test_static_ascend310p3)
ut_case.add_cust_test_func(test_func=test_static_lhisi)

# vals = {("tik.vbitsort32"): True}
def side_effects(*args):
    # return vals[args]
    return True

def test_v220_mock(test_arg):
    #with patch("te.platform.cce_conf.api_check_support",MagicMock(side_effect=side_effects)):
    with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
        with patch("te.tik.Tik.vmrgsort",MagicMock(side_effect=side_effects)):
            with patch("te.tik.Tik.vadds",MagicMock(side_effect=side_effects)):
                with patch("te.tik.Tik.vsort32",MagicMock(side_effect=side_effects)):
                    with patch("te.tik.Tik.vreduce",MagicMock(side_effect=side_effects)):
                        from impl.top_k_d import top_k_d
                        top_k_d({"shape": (100000,), "format": "ND", "dtype": "float16", "ori_shape": (100000,), "ori_format": "ND"},
                                {"shape": (8192, ), "format": "ND", "dtype": "float16", "ori_shape": (8192,), "ori_format": "ND"},
                                {"shape": (100000,), "format": "ND", "dtype": "float16", "ori_shape": (100000,), "ori_format": "ND"},
                                {"shape": (100000,), "format": "ND", "dtype": "int32", "ori_shape": (100000,), "ori_format": "ND"},
                                100000, True, -1, True)

                        top_k_d({"shape": (1, 10000), "format": "ND", "dtype": "float16", "ori_shape": (1, 10000), "ori_format": "ND"},
                                {"shape": (2048,), "format": "ND", "dtype": "int32", "ori_shape": (2048,), "ori_format": "ND"},
                                {"shape": (1, 5000), "format": "ND", "dtype": "float16", "ori_shape": (1, 5000), "ori_format": "ND"},
                                {"shape": (1, 5000), "format": "ND", "dtype": "int32", "ori_shape": (1, 5000), "ori_format": "ND"},
                                5000, True, -1, True)

                        top_k_d({"shape": (5, 10000), "format": "ND", "dtype": "float16", "ori_shape": (5, 10000), "ori_format": "ND"},
                                {"shape": (2048,), "format": "ND", "dtype": "int32", "ori_shape": (2048,), "ori_format": "ND"},
                                {"shape": (5, 5000), "format": "ND", "dtype": "float16", "ori_shape": (5, 5000), "ori_format": "ND"},
                                {"shape": (5, 5000), "format": "ND", "dtype": "int32", "ori_shape": (5, 5000), "ori_format": "ND"},
                                5000, True, -1, True)
ut_case.add_cust_test_func(test_func=test_v220_mock)
