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

TabulateFusionGrad ut case
"""
from op_test_frame.ut import OpUT
from impl.dynamic.tabulate_fusion_grad import tabulate_fusion_grad
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("TabulateFusionGrad", "impl.dynamic.tabulate_fusion_grad", "tabulate_fusion_grad")


def tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0, split_count=1, split_index=0):
    set_current_compile_soc_info("Ascend710")

    with tbe.common.context.op_context.OpContext("dynamic"):
        table = {"shape": (table_dim0, last_layer_size*6), "dtype": "float32", "format": "ND",
                 "ori_shape": (table_dim0, last_layer_size*6), "ori_format": "ND",
                 "range": ((table_dim0, table_dim0), (last_layer_size*6, last_layer_size*6))}
        table_info = {"shape": (6,), "dtype": "float32", "format": "ND",
                      "ori_shape": (6,), "ori_format": "ND",
                      "range": ((6, 6))}
        em_x = {"shape": (nloc * nnei, 1), "dtype": "float32", "format": "ND",
                "ori_shape": (nloc * nnei, 1), "ori_format": "ND",
                "range": ((nloc * nnei, nloc * nnei), (1, 1))}
        em = {"shape": (nloc, nnei, 4), "dtype": "float32", "format": "ND",
              "ori_shape": (nloc, nnei, 4), "ori_format": "ND",
              "range": ((nloc, nloc), (nnei, nnei), (4, 4))}
        dy = {"shape": (nloc, 4, last_layer_size), "dtype": "float32", "format": "ND",
              "ori_shape": (nloc, 4, last_layer_size), "ori_format": "ND",
              "range": ((nloc, nloc), (4, 4), (last_layer_size, last_layer_size))}
        descriptor = {"shape": (nloc, 4, last_layer_size), "dtype": "float32", "format": "ND",
                      "ori_shape": (nloc, 4, last_layer_size), "ori_format": "ND",
                      "range": ((nloc, nloc), (4, 4), (last_layer_size, last_layer_size))}
        dy_dem_x = {"shape": (nloc * nnei, 1), "dtype": "float32", "format": "ND",
                    "ori_shape": (nloc * nnei, 1), "ori_format": "ND",
                    "range": ((nloc * nnei, nloc * nnei), (1, 1))}
        dy_dem = {"shape": (nloc, nnei, 4), "dtype": "float32", "format": "ND",
                  "ori_shape": (nloc, nnei, 4), "ori_format": "ND",
                  "range": ((nloc, nloc), (nnei, nnei), (4, 4))}

        tabulate_fusion_grad(table, table_info, em_x, em, dy, descriptor, dy_dem_x, dy_dem,
                             split_count, split_index, kernel_name="tabulate_fusion_grad_ut")

    set_current_compile_soc_info(test_args)


def test_tabulate_fusion_grad_case001(test_args):
    nloc, nnei, last_layer_size, table_dim0 = 4096, 46, 100, 1360
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0)
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0, 2, 0)
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0, 2, 1)

def test_tabulate_fusion_grad_case002(test_args):
    nloc, nnei, last_layer_size, table_dim0 = 8192, 92, 128, 1360
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0)
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0, 2, 0)
    tabulate_fusion_grad_ut_test(test_args, nloc, nnei, last_layer_size, table_dim0, 2, 1)


ut_case.add_cust_test_func(test_func=test_tabulate_fusion_grad_case001)
ut_case.add_cust_test_func(test_func=test_tabulate_fusion_grad_case002)


if __name__ == '__main__':
    ut_case.run("Ascend710")
    exit(0)
