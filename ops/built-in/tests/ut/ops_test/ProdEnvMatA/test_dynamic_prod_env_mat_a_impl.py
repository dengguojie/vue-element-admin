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
from impl.dynamic.prod_env_mat_a import prod_env_mat_a
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("ProdEnvMatA", "impl.dynamic.prod_env_mat_a",
               "prod_env_mat_a")

def simple_water_test(test_args, nsample, nloc, n_a_sel, n_r_sel, nall,
                      rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, split_count=1, split_index=0, kernel_name = "prod_env_mat_a"):
    nnei = n_a_sel + n_r_sel

    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_env_mat_a({"shape": (nsample, nall * 3), "dtype": "float32", "format": "ND",
                        "ori_shape": (nsample, nall * 3), "ori_format": "ND",
                        "range": ((nsample, nsample), (nall * 3, nall * 3))},
                       {"shape": (nsample, nall), "dtype": "int32", "format": "ND",
                        "ori_shape": (nsample, nall), "ori_format": "ND",
                        "range": ((nsample, nsample), (nall, nall))},
                       {"shape": [4], "dtype": "int32", "format": "ND",
                        "ori_shape": [4], "ori_format": "ND",
                        "range": ((4, 4))},
                       {"shape": (nsample, 9), "dtype": "float32", "format": "ND",
                        "ori_shape": (nsample, 9), "ori_format": "ND",
                        "range": ((nsample, nsample), (9, 9))},

                       {"shape": (1 + 1026 * nloc,), "dtype": "int32", "format": "ND",
                        "ori_shape": (1 + 1026 * nloc,), "ori_format": "ND", "range": ((1 + 1026 * nloc,
                                                                                        1 + 1026 * nloc))},
                       {"shape": (2, nnei * 4), "dtype": "float32", "format": "ND",
                        "ori_shape": (2, nnei * 4),
                        "ori_format": "ND", "range": ((2, 2), (nnei * 4, nnei * 4))},
                       {"shape": (2, nnei * 4), "dtype": "float32", "format": "ND",
                        "ori_shape": (2, nnei * 4),
                        "ori_format": "ND", "range": ((2, 2), (nnei * 4, nnei * 4))},

                       {"shape": (nsample, nloc * nnei * 4), "dtype": "float32", "format": "ND",
                        "ori_shape": (nsample, nloc * nnei * 4), "ori_format": "ND",
                        "range": ((nsample, nsample), (nloc * nnei * 4, nloc * nnei * 4))},
                       {"shape": (nsample, nloc * nnei * 12), "dtype": "float32", "format": "ND",
                        "ori_shape": (nsample, nloc * nnei * 12), "ori_format": "ND",
                        "range": ((nsample, nsample), (nloc * nnei * 12, nloc * nnei * 12))},
                       {"shape": (nsample, nloc * nnei * 3), "dtype": "float32", "format": "ND",
                        "ori_shape": (nsample, nloc * nnei * 3), "ori_format": "ND",
                        "range": ((nsample, nsample), (nloc * nnei * 3, nloc * nnei * 3))},
                       {"shape": (nsample, nloc * nnei), "dtype": "int32", "format": "ND",
                        "ori_shape": (nsample, nloc * nnei), "ori_format": "ND",
                        "range": ((nsample, nsample), (nloc * nnei, nloc * nnei))},
                         rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, split_count, split_index, kernel_name)
    set_current_compile_soc_info(test_args)


def test_prod_env_mat_a_case001(test_args):
    nframes = 1
    nloc = 12288
    n_a_sel = 138
    n_r_sel = 0
    nall = 28328
    rcut_a = 0.0
    rcut_r = 8.0
    rcut_r_smth = 2.0
    sel_a = [46, 92]
    sel_r = []
    kernel_name = "prod_env_mat_a"
    split_count = 1
    split_index = 0

    simple_water_test(test_args, nframes, nloc, n_a_sel, n_r_sel, nall, rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r, split_count, split_index, kernel_name)


ut_case.add_cust_test_func(test_func=test_prod_env_mat_a_case001)

if __name__ == '__main__':
    ut_case.run("Ascend710")
    exit(0)
