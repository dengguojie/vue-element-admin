#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from impl.dynamic.cast import cast
from te import platform as cce_conf
import tbe

ut_case = OpUT("Cast", "impl.dynamic.cast", "cast")

case1 = {"params": [
    {"shape": (-1, -1, 2), "ori_shape": (2, 10, 5), "range": ((1, None), (1, None), (2, 2)), "format": "NHWC",
     "ori_format": "NHWC", 'dtype': "float32"},
    {"shape": (-1, -1, 2), "ori_shape": (2, 10, 5), "range": ((1, None), (1, None), (2, 2)), "format": "NHWC",
     "ori_format": "NHWC", 'dtype': "float16"},
    1],
         "case_name": "cast_dynamic_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1,), "ori_shape": (100,), "range": ((1, None),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "float32"},
                    {"shape": (1,), "ori_shape": (1,), "range": ((1, 1),), "format": "NHWC", "ori_format": "NHWC",
                     'dtype': "int32"},
                    3],
         "case_name": "cast_dynamic_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float16"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float32"},
    0],
         "case_name": "cast_dynamic_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float32"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int32"},
    3],
         "case_name": "cast_dynamic_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int8"},
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float32"},
    0],
         "case_name": "cast_dynamic_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [
    {"shape": (-1, -1), "ori_shape": (2, 2), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"},
    {"shape": (-1, -1), "ori_shape": (2, 2), "range": ((2, 2), (2, 2)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"},
    1],
         "case_name": "cast_dynamic_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [
    {"shape": (-1, -1), "ori_shape": (2, 2), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "uint8"},
    {"shape": (-1, -1), "ori_shape": (2, 2), "range": ((2, 2), (2, 2)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int32"},
    3],
         "case_name": "cast_dynamic_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "int32"},
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "uint8"},
    4],
         "case_name": "cast_dynamic_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "float16"},
    {"shape": (-1, -1), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
     'dtype': "bool"},
    12],
         "case_name": "cast_dynamic_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)


def ascend920A_001(test_args):
    cce_conf.cce_conf.te_set_version("Ascend920A", "VectorCore")
    with tbe.common.context.op_context.OpContext("dynamic"):
        cast({"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int64"},
             {"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "float32"},
             0)
    cce_conf.cce_conf.te_set_version(test_args)


def ascend920A_002(test_args):
    cce_conf.cce_conf.te_set_version("Ascend920A", "VectorCore")
    with tbe.common.context.op_context.OpContext("dynamic"):
        cast({"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "float32"},
             {"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int64"},
             9)
    cce_conf.cce_conf.te_set_version(test_args)


def ascend920A_003(test_args):
    cce_conf.cce_conf.te_set_version("Ascend920A", "VectorCore")
    with tbe.common.context.op_context.OpContext("dynamic"):
        cast({"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int64"},
             {"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int32"},
             3)
    cce_conf.cce_conf.te_set_version(test_args)


def ascend920A_004(test_args):
    cce_conf.cce_conf.te_set_version("Ascend920A", "VectorCore")
    with tbe.common.context.op_context.OpContext("dynamic"):
        cast({"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int32"},
             {"shape": (2, 5), "ori_shape": (2, 5), "range": ((1, None), (1, None)), "format": "NHWC", "ori_format": "NHWC",
              'dtype': "int64"},
             9)
    cce_conf.cce_conf.te_set_version(test_args)


ut_case.add_cust_test_func(test_func=ascend920A_001)
ut_case.add_cust_test_func(test_func=ascend920A_002)
ut_case.add_cust_test_func(test_func=ascend920A_003)
ut_case.add_cust_test_func(test_func=ascend920A_004)

if __name__ == '__main__':
    ut_case.run("Ascend910")
