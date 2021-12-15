#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ProdVirialSeA ut case
"""
from op_test_frame.ut import OpUT
from impl.dynamic.prod_virial_se_a import prod_virial_se_a
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("ProdVirialSeA", "impl.dynamic.prod_virial_se_a",
               "prod_virial_se_a")


def simple_water_test(test_args, nframes: int, nloc: int, n_a_sel: int, n_r_sel: int, nall: int, natomsSize: int,
                      split_count=1, split_index=0):
    nnei = n_a_sel + n_r_sel

    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_virial_se_a({"shape": (nframes, nloc * nnei * 4), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, nloc * nnei * 4), "ori_format": "ND",
                          "range": ((nframes, nframes), (nloc * nnei * 4, nloc * nnei * 4))},
                         {"shape": (nframes, nloc * nnei * 4 * 3), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, nloc * nnei * 4 * 3), "ori_format": "ND",
                          "range": ((nframes, nframes), (nloc * nnei * 4 * 3, nloc * nnei * 4 * 3))},
                         {"shape": (nframes, nloc * nnei * 3), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, nloc * nnei * 3), "ori_format": "ND",
                          "range": ((nframes, nframes), (nloc * nnei * 3, nloc * nnei * 3))},
                         {"shape": (nframes, nloc * nnei), "dtype": "int32", "format": "ND",
                          "ori_shape": (nframes, nloc * nnei), "ori_format": "ND",
                          "range": ((nframes, nframes), (nloc * nnei, nloc * nnei))},
                         {"shape": (natomsSize,), "dtype": "int32", "format": "ND",
                          "ori_shape": (natomsSize,), "ori_format": "ND", "range": ((natomsSize, natomsSize))},
                         {"shape": (nframes, 9), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, 9), "ori_format": "ND", "range": ((nframes, nframes), (9, 9))},
                         {"shape": (nframes, nall * 9), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, nall * 9), "ori_format": "ND",
                          "range": ((nframes, nframes), (nall * 9, nall * 9))},
                         nnei, 0, split_count, split_index)
    set_current_compile_soc_info(test_args)


def test_prod_virial_se_a_case001(test_args):
    nframes = 1
    nloc = 12288
    n_a_sel = 138
    n_r_sel = 0
    nall = 28328
    natomsSize = 4

    simple_water_test(test_args, nframes, nloc, n_a_sel, n_r_sel, nall, natomsSize)
    simple_water_test(test_args, nframes, nloc, n_a_sel, n_r_sel, nall, natomsSize, 2, 0)
    simple_water_test(test_args, nframes, nloc, n_a_sel, n_r_sel, nall, natomsSize, 2, 1)


ut_case.add_cust_test_func(test_func=test_prod_virial_se_a_case001)

if __name__ == '__main__':
    ut_case.run("Ascend710")
    exit(0)
