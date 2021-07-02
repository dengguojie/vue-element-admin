#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of onehot
"""

from op_test_frame.ut import OpUT

ut_case = OpUT("EmbeddingDenseGrad", "impl.dynamic.embedding_dense_grad", "embedding_dense_grad")

case1 = {"params": [
    {"shape": (-1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 20000, -1, False],
         "case_name": "dynamic_embedding_dense_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 30000, -1, False],
         "case_name": "dynamic_embedding_dense_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 30000, -1, False],
         "case_name": "dynamic_embedding_dense_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)
        , (2, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 10000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)
        , (2, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 10000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 10000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 10000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 40000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                                -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 40000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 40000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 30001, 2, False],
         "case_name": "dynamic_embedding_dense_grad_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1, -1), "dtype": "float32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 40000, 2, False],
         "case_name": "dynamic_embedding_dense_grad_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case_lis = [case1, case2, case3, case4, case5, case6, case7, case8, case9, case10, case11, case12]
for ele in case_lis:
    ut_case.add_case("Ascend910A", ele)
if __name__ == '__main__':
    ut_case.run("Ascend910A")

