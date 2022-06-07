#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ProdVirialSeA st case
"""
import tbe
from impl.dynamic.tabulate_fusion import tabulate_fusion
from tbe.common.platform.platform_info import set_current_compile_soc_info


def tabulate_fusion_test(nloc, nnei, last_layer_size, table_dim0, split_count=1, split_index=0, soc="310P3"):
    if soc == "910":
        set_current_compile_soc_info("Ascend910")
    else:
        set_current_compile_soc_info("Ascend310P3")

    last_layer_size_align = (last_layer_size + 64 - 1) // 64 * 64
    with tbe.common.context.op_context.OpContext("dynamic"):
        table = {"shape": (table_dim0, last_layer_size_align*6), "dtype": "float32", "format": "ND",
                 "ori_shape": (table_dim0, last_layer_size_align*6), "ori_format": "ND",
                 "range": ((table_dim0, table_dim0), (last_layer_size_align*6, last_layer_size_align*6))}
        table_info = {"shape": (6,), "dtype": "float32", "format": "ND",
                      "ori_shape": (6,), "ori_format": "ND",
                      "range": ((6, 6))}
        em_x = {"shape": (nloc * nnei, 1), "dtype": "float32", "format": "ND",
                "ori_shape": (nloc * nnei, 1), "ori_format": "ND",
                "range": ((nloc * nnei, nloc * nnei), (1, 1))}
        em = {"shape": (nloc, nnei, 4), "dtype": "float32", "format": "ND",
              "ori_shape": (nloc, nnei, 4), "ori_format": "ND",
              "range": ((nloc, nloc), (nnei, nnei), (4, 4))}
        descriptor = {"shape": (nloc, 4, last_layer_size), "dtype": "float32", "format": "ND",
                      "ori_shape": (nloc, 4, last_layer_size), "ori_format": "ND",
                      "range": ((nloc, nloc), (4, 4), (last_layer_size, last_layer_size))}
        tabulate_fusion(table, table_info, em_x, em, descriptor, last_layer_size, split_count, split_index,
                        kernel_name="tabulate_fusion_ut")


def test_tabulate_fusion_case001(nloc, nnei, last_layer_size, table_dim0):
    tabulate_fusion_test(nloc, nnei, last_layer_size, table_dim0)
    tabulate_fusion_test(nloc, nnei, last_layer_size, table_dim0, 2, 0)
    tabulate_fusion_test(nloc, nnei, last_layer_size, table_dim0, 2, 1)
    tabulate_fusion_test(nloc, nnei, last_layer_size, table_dim0, soc="910")


if __name__ == '__main__':
    test_tabulate_fusion_case001(nloc=4096, nnei=46, last_layer_size=100, table_dim0=1360)
    test_tabulate_fusion_case001(nloc=8192, nnei=92, last_layer_size=128, table_dim0=1360)
    exit(0)
