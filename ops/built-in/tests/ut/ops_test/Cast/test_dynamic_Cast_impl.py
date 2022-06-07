#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

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

case10 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float16"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float32"},
    None],
         "case_name": "cast_dynamic_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "float16"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int8"},
    2],
         "case_name": "cast_dynamic_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910A"], case11)

case12 = {"params": [
    {"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int32"},
    {"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int64"},
    9],
         "case_name": "cast_dynamic_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int32"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int64"},
    9],
         "case_name": "cast_dynamic_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case14 = {"params": [
    {"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int64"},
    {"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int32"},
    3],
         "case_name": "cast_dynamic_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case15 = {"params": [
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int64"},
    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND",
     'dtype': "int32"},
    3],
         "case_name": "cast_dynamic_15",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310P3", "Ascend910A"], case12)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case13)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case14)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case15)

def test_check_supported(test_arg):
    from impl.dynamic.cast import check_supported
    check_supported({"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND", 'dtype': "int32"},
                    {"shape": (200, 5), "ori_shape": (200, 5), "range": ((1, 100), (1, 5)), "format": "ND", "ori_format": "ND", 'dtype': "int64"}, 9)

    check_supported({"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND", 'dtype': "float32"},
                    {"shape": (-1, 5), "ori_shape": (200, 5), "range": ((1, None), (1, 5)), "format": "ND", "ori_format": "ND", 'dtype': "float16"}, 1)

ut_case.add_cust_test_func(test_func=test_check_supported)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
