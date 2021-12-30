#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
ut of onehot
"""

from op_test_frame.ut import OpUT

ut_case = OpUT("OneHot", "impl.dynamic.one_hot", "one_hot")

case1 = {"params": [
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]}, -1],
         "case_name": "dynamic_one_hot_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]}, 0],
         "case_name": "dynamic_one_hot_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]}, 2],
         "case_name": "dynamic_one_hot_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 0],
         "case_name": "dynamic_one_hot_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]}, 0], 
         "case_name": "dynamic_one_hot_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]}, 0],
         "case_name": "dynamic_one_hot_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]}, 0],
         "case_name": "dynamic_one_hot_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1, ),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]}, 0],
         "case_name": "dynamic_one_hot_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None), (2, None)]}, -1],
         "case_name": "dynamic_one_hot_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None), (2, None)]}, 3],
         "case_name": "dynamic_one_hot_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None), (2, None)]}, 2],
         "case_name": "dynamic_one_hot_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]}, 2],
         "case_name": "dynamic_one_hot_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]}, 1],
         "case_name": "dynamic_one_hot_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case14 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]}, 3],
         "case_name": "dynamic_one_hot_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case15 = {"params": [
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None)]}, -1],
         "case_name": "dynamic_one_hot_15",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case16 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]}, 1],
         "case_name": "dynamic_one_hot_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case17 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]}, 2],
         "case_name": "dynamic_one_hot_17",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case18 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]}, -1],
         "case_name": "dynamic_one_hot_18",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case19 = {"params": [
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None)]}, 3],
         "case_name": "dynamic_one_hot_19",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case20 = {"params": [
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]}, 1],
         "case_name": "dynamic_one_hot_20",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case21 = {"params": [
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]}, 2],
         "case_name": "dynamic_one_hot_21",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case22 = {"params": [
    {"shape": (-1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]}, -1],
         "case_name": "dynamic_one_hot_22",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case23 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 0],
         "case_name": "dynamic_one_hot_23",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case24 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 1],
         "case_name": "dynamic_one_hot_24",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case25 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 2],
         "case_name": "dynamic_one_hot_25",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case26 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, -3],
         "case_name": "dynamic_one_hot_26",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case27 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 4],
         "case_name": "dynamic_one_hot_27",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case28 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 5],
         "case_name": "dynamic_one_hot_28",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case29 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, -1],
         "case_name": "dynamic_one_hot_29",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case30 = {"params": [
    {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1, -1,
                                                                                            -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1, -1, -1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1,
                                                                                                -1, -1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None), (2, None), (2, None), (2, None),
                                   (2, None)]}, 6], 
         "case_name": "dynamic_one_hot_30",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case31 = {"params": [
    {"shape": (2, 2), "dtype": "int32", "format": "ND", "ori_shape": (2, 2),
     "ori_format": "ND", "range": [(2, None), (2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(2, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]},
    {"shape": (-1,), "dtype": "int32", "format": "ND", "ori_shape": (-1,),
     "ori_format": "ND", "range": [(1, None)]}, 
    {"shape": (-1, -1, -1), "dtype": "int32", "format": "ND", "ori_shape": (-1, -1, -1),
     "ori_format": "ND", "range": [(2, None), (2, None), (2, None)]}, 0],
         "case_name": "dynamic_one_hot_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
def test_check_supported(test_arg):
    from impl.dynamic.one_hot import check_supported
    check_supported({"shape": (2048,), "dtype": "float16", "format": "NCHW", "ori_shape": (2048,),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (2048,2), "dtype": "float32", "format": "NCHW", "ori_shape": (2048,2),"ori_format": "NCHW"},-1)
    check_supported({"shape": (2048,), "dtype": "int32", "format": "NCHW", "ori_shape": (2048,),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (2048,2), "dtype": "float32", "format": "NCHW", "ori_shape": (2048,2),"ori_format": "NCHW"},-1)
    check_supported({"shape": (2048,), "dtype": "int32", "format": "NCHW", "ori_shape": (2048,),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "int8", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (2048,2), "dtype": "int8", "format": "NCHW", "ori_shape": (2048,2),"ori_format": "NCHW"},-1)
    check_supported({"shape": (-1,2048), "dtype": "int32", "format": "NCHW", "ori_shape": (-1,2048),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (-1,2048,-1), "dtype": "float32", "format": "NCHW", "ori_shape": (-1,2048,-1),"ori_format": "NCHW"},-1)
    check_supported({"shape": (2048,), "dtype": "int32", "format": "NCHW", "ori_shape": (2048,),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (2,2048,), "dtype": "float32", "format": "NCHW", "ori_shape": (2,2048),"ori_format": "NCHW"},0)
    check_supported({"shape": (2048,), "dtype": "int32", "format": "NCHW", "ori_shape": (2048,),"ori_format": "NCHW"},
                     {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,),"ori_format": "ND"},
                     {"shape": (2048,2), "dtype": "float32", "format": "NCHW", "ori_shape": (2048,2),"ori_format": "NCHW"},-1)
ut_case.add_cust_test_func(test_func=test_check_supported)
ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)
ut_case.add_case("all", case7)
ut_case.add_case("all", case8)
ut_case.add_case("all", case9)
ut_case.add_case("all", case10)
ut_case.add_case("all", case11)
ut_case.add_case("all", case12)
ut_case.add_case("all", case13)
ut_case.add_case("all", case14)
ut_case.add_case("all", case15)
ut_case.add_case("all", case16)
ut_case.add_case("all", case17)
ut_case.add_case("all", case18)
ut_case.add_case("all", case19)
ut_case.add_case("all", case20)
ut_case.add_case("all", case21)
ut_case.add_case("all", case22)
ut_case.add_case("all", case23)
ut_case.add_case("all", case24)
ut_case.add_case("all", case25)
ut_case.add_case("all", case26)
ut_case.add_case("all", case27)
ut_case.add_case("all", case28)
ut_case.add_case("all", case29)
ut_case.add_case("all", case30)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
