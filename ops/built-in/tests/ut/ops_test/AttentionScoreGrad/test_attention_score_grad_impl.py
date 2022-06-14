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

AttentionScore ut case
"""
from op_test_frame.ut import OpUT
from impl.attention_score_grad import attention_score_grad
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("AttentionScoreGrad", "impl.attention_score_grad",
               "attention_score_grad")

def test_train(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, kernel_name):
    set_current_compile_soc_info("Ascend910")
    attention_score_grad({"shape": (batch_dim0, batch_dim1, seq_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                         {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                         {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                         {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                         {"shape": (batch_dim0, batch_dim1, n_num, seq_num, nz_dim, nz_dim), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, n_num * nz_dim), "ori_format": "ND"},
                         {"shape": (1,), "dtype": "float16",
                          "format": "ND",
                          "ori_shape": (1,), "ori_format": "ND"},
                         {"shape": (batch_dim0, batch_dim1, seq_num, seq_num, nz_dim, nz_dim), "dtype": "uint8",
                          "format": "ND",
                          "ori_shape": (batch_dim0, batch_dim1, seq_num * nz_dim, seq_num * nz_dim),
                          "ori_format": "ND"},
                         {"shape": (64, 768, 16, 16), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (12288, 1024), "ori_format": "ND"},
                         {"shape": (64, 768, 16, 16), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (12288, 1024), "ori_format": "ND"},
                         {"shape": (64, 768, 16, 16), "dtype": "float16",
                          "format": "FRACTAL_NZ",
                          "ori_shape": (12288, 1024), "ori_format": "ND"},
                         1.0, False, False, False, False, [-1], kernel_name)

def test_attention_score_case001(test_args):

    kernel_name = "attention_socre_grad"
    batch_dim0 = 32
    batch_dim1 = 16
    seq_num = 24
    n_num = 4
    nz_dim = 16

    test_train(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, kernel_name)

    kernel_name = "attention_socre_grad_01"
    batch_dim0 = 32
    batch_dim1 = 16
    seq_num = 8
    n_num = 4
    nz_dim = 16

    test_train(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, kernel_name)

    kernel_name = "attention_socre_grad_02"
    batch_dim0 = 32
    batch_dim1 = 16
    seq_num = 16
    n_num = 4
    nz_dim = 16

    test_train(batch_dim0, batch_dim1, seq_num, n_num, nz_dim, kernel_name)


ut_case.add_cust_test_func(test_func=test_attention_score_case001)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
