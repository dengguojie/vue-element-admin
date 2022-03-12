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
from impl.dynamic.prod_virial_se_a import prod_virial_se_a
from tbe.common.platform.platform_info import set_current_compile_soc_info


def simple_test(nframes: int, nloc: int, n_a_sel: int, n_r_sel: int, nall: int, natoms_size: int,
                split_count=1, split_index=0):
    nnei = n_a_sel + n_r_sel

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
                         {"shape": (natoms_size,), "dtype": "int32", "format": "ND",
                          "ori_shape": (natoms_size,), "ori_format": "ND", "range": ((natoms_size, natoms_size))},
                         {"shape": (nframes, 9), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, 9), "ori_format": "ND", "range": ((nframes, nframes), (9, 9))},
                         {"shape": (nframes, nall * 9), "dtype": "float32", "format": "ND",
                          "ori_shape": (nframes, nall * 9), "ori_format": "ND",
                          "range": ((nframes, nframes), (nall * 9, nall * 9))},
                         n_a_sel, n_r_sel, split_count, split_index)


def test_prod_virial_se_a_case(
        nframes,
        nloc,
        n_a_sel,
        n_r_sel,
        nall,
        natoms_size):

    set_current_compile_soc_info("Ascend710")
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size)
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size, 2, 0)
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size, 2, 1)

    set_current_compile_soc_info("Ascend910")
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size)
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size, 2, 0)
    simple_test(nframes, nloc, n_a_sel, n_r_sel, nall, natoms_size, 2, 1)


if __name__ == '__main__':
    test_prod_virial_se_a_case(nframes=1,
                               nloc=12288,
                               n_a_sel=138,
                               n_r_sel=0,
                               nall=28328,
                               natoms_size=4)

    test_prod_virial_se_a_case(nframes=1,
                               nloc=13500,
                               n_a_sel=138,
                               n_r_sel=0,
                               nall=30000,
                               natoms_size=4)

    exit(0)
