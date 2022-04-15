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

AttentionScore st case
"""
from impl.attention_score import attention_score
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe

def test_tuili(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, drop_mask_dtype, kernel_name):
    attention_score({"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                    "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                    {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                     "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                    {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                     "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                    {"shape": (batch_dim0, 1, seq_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                     "ori_shape": (batch_dim0, 1, seq_num * nz_dim, seq_num * nz_dim), "ori_format": "ND"},
                    {"shape": (1,), "dtype": "float16",
                     "format": "ND",
                     "ori_shape": (1,), "ori_format": "ND"},
                    {"shape": (batch_dim0, batch_dim1, seq_num, seq_num, nz_dim, nz_dim), "dtype": drop_mask_dtype,
                     "format": "ND",
                     "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, seq_num * nz_dim), "ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                     "ori_shape": (12288, 1024), "ori_format": "ND"},
                    {"shape": (batch_dim0, batch_dim1, seq_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                     "format": "FRACTAL_NZ",
                     "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, seq_num * nz_dim), "ori_format": "ND"},
                    1.0, False, False, False, False, [-1], kernel_name)


def test_attention_score_case():

    kernel_name = "attention_socre"
    batch_dim0 = 32
    batch_dim1 = 16
    seq_num = 24
    n_num = 4
    nz_dim = 16
    set_current_compile_soc_info("Ascend710")
    drop_mask_dtype = "float16"
    test_tuili(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, drop_mask_dtype, kernel_name)

    set_current_compile_soc_info("Ascend910")
    drop_mask_dtype = "uint8"
    test_tuili(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, drop_mask_dtype, kernel_name)


if __name__ == '__main__':
    test_attention_score_case()

    exit(0)
