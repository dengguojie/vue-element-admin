#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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

ProdForceSeA ut case
"""
from op_test_frame.ut import OpUT
from impl.prod_force_se_a import prod_force_se_a
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("ProdForceSeA", "impl.prod_force_se_a", "prod_force_se_a")


def test_prod_force_se_a_case001(test_args):
    """
    Compute prod_force_se_a.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_force_se_a({"shape": (1, 6782976), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 6782976), "ori_format": "ND", "range": ((1, 1), (6782976, 6782976))},
                {"shape": (1, 20348928), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 20348928), "ori_format": "ND", "range": ((1, 1), (20348928, 20348928))},
                {"shape": (1, 1695744), "dtype": "int32", "format": "ND",
                "ori_shape": (1, 1695744), "ori_format": "ND", "range": ((1, 1), (1695744, 1695744))},
                {"shape": (3,), "dtype": "int32", "format": "ND",
                "ori_shape": (3,), "ori_format": "ND", "range": ((3, 3))},
                {"shape": (1, 84984), "dtype": "float32", "format": "ND",
                "ori_shape": (1, 84984), "ori_format": "ND", "range": ((1, 1), (84984, 84984))},
                138, 0)
    set_current_compile_soc_info(test_args)


ut_case.add_cust_test_func(test_func=test_prod_force_se_a_case001)


if __name__ == '__main__':
    ut_case.run("Ascend710")
    exit(0)
