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

ProdEnvMatACalcDescrpt st case
"""
import tbe
from impl.dynamic.prod_env_mat_a_calc_descrpt import prod_env_mat_a_calc_descrpt
from tbe.common.platform.platform_info import set_current_compile_soc_info


def simple_water_test_910(nsample, nloc, nall, n_a_sel, n_r_sel, rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r):
    nnei = n_a_sel + n_r_sel

    with tbe.common.context.op_context.OpContext("dynamic"):
        prod_env_mat_a_calc_descrpt({"shape": (nsample, nloc * nnei), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei, nloc * nnei))},
                                    {"shape": (nsample, nloc * nnei), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei, nloc * nnei))},
                                    {"shape": (nsample, nloc * nnei), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei, nloc * nnei))},
                                    {"shape": (nsample, nloc * nnei), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei, nloc * nnei))},
                                    {"shape": (nsample, nall), "dtype": "int32", "format": "ND",
                                     "ori_shape": (nsample, nall), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nall, nall))},
                                    {"shape": (4,), "dtype": "int32", "format": "ND",
                                     "ori_shape": (4,), "ori_format": "ND", "range": ((4, 4))},
                                    {"shape": (1 + 1026 * nloc,), "dtype": "int32", "format": "ND",
                                     "ori_shape": (1 + 1026 * nloc,), "ori_format": "ND",
                                     "range": ((1 + 1026 * nloc, 1 + 1026 * nloc))},
                                    {"shape": (2, nnei * 4), "dtype": "float32", "format": "ND",
                                     "ori_shape": (2, nnei * 4), "ori_format": "ND",
                                     "range": ((2, 2), (nnei * 4, nnei * 4))},
                                    {"shape": (2, nnei * 4), "dtype": "float32", "format": "ND",
                                     "ori_shape": (2, nnei * 4), "ori_format": "ND",
                                     "range": ((2, 2), (nnei * 4, nnei * 4))},
                                    {"shape": (nsample, nloc * nnei * 4), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei * 4), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei * 4, nloc * nnei * 4))},
                                    {"shape": (nsample, nloc * nnei * 12), "dtype": "float32", "format": "ND",
                                     "ori_shape": (nsample, nloc * nnei * 12), "ori_format": "ND",
                                     "range": ((nsample, nsample), (nloc * nnei * 12, nloc * nnei * 12))},
                                    rcut_a, rcut_r, rcut_r_smth, sel_a, sel_r)


def test_prod_env_mat_a_calc_descrpt_case001():
    set_current_compile_soc_info("Ascend910")
    simple_water_test_910(1, 12288, 28328, 138, 0, 0.0, 8.0, 2.0, [46, 92], [])


if __name__ == '__main__':
    test_prod_env_mat_a_calc_descrpt_case001()
    exit(0)
