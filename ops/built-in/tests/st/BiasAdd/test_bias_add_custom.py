#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("BiasAdd", "impl.dynamic.bias_add", "bias_add")

case1 = {"params": [
        {"ori_shape": (-1,1, -1,1,16), "shape": (-1,1, -1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((1, 100),(1, 1,),(1, 100),(1, 100),(16,16))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "NDHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((1, 1),)},
        {"ori_shape": (-1,1, -1,1,16), "shape": (-1,1, -1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((1, 100),(1, 1,),(1, 100),(1, 100),(16,16))}],
         "case_name": "bias_add_case_error_1",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case2 = {"params": [
        {"ori_shape": (-1,16, -1,1,16), "shape": (-1,1, -1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((1, 100),(1, 1,),(1, 100),(1, 100),(16,16))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "NCDHW",
         "format": "NCDHW", "dtype": "float16", "range": ((1, 1),)},
        {"ori_shape": (-1,1, -1,1,16), "shape": (-1,1, -1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((1, 100),(1, 1,),(1, 100),(1, 100),(16,16))}],
         "case_name": "bias_add_case_error_2",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case3 = {"params": [
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))},
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))}],
         "case_name": "bias_add_case_error_3",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case4 = {"params": [
        {"ori_shape": (-1,1,16), "shape": (-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (-1,1,16), "shape": (-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_4",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case5 = {"params": [
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))}],
         "case_name": "bias_add_case_error_5",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case6 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_6",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case7 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_7",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case8 = {"params": [
        {"ori_shape": (10,1,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NC1HWC0",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NC1HWC0",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_8",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case9 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NCHW",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "NCHW",
         "format": "NC1HWC0", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NCHW",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_9",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case10 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_10",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case11 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NCHW",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NCHW",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_11",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case12 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NC1HWC0", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_12",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case13 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_13",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case14 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_14",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case15 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((10, 10,),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_15",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case16 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_16",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case17 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_17",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case18 = {"params": [
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (10,1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NHWC",
         "format": "NCDHW", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_18",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case19 = {"params": [
        {"ori_shape": (-1,1,-1,1,16), "shape": (1,-1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "NDHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,-1,-1,-1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_19",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case20 = {"params": [
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "NDHWC",
         "format": "NDHWC", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,-1,16), "ori_format": "NDHWC",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_20",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case21 = {"params": [
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,1,16), "ori_format": "NCDHW",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,-1,16), "ori_format": "NCDHW",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_21",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case22 = {"params": [
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,1,16), "ori_format": "NCDHW",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (-1,1,-1,1,16), "shape": (-1,1,-1,-1,-1,16), "ori_format": "NCDHW",
         "format": "NDC1HWC0", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_22",
         "expect": "success", "format_expect": [],
         "support_expect": True}

case23 = {"params": [
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))},
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (16,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16))}],
         "case_name": "bias_add_case_error_23",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case24 = {"params": [
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_24",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case25 = {"params": [
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1,1),)},
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_25",
         "expect": "success", "format_expect": [],
         "support_expect": True}

case26 = {"params": [
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))},
        {"ori_shape": (16,), "shape": (16,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((16,16),)},
        {"ori_shape": (1,1,1,16), "shape": (-1,-1,1,16), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((10,10),(1, 1,),(1, 1,),(16,16))}],
         "case_name": "bias_add_case_error_26",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend910A"], case12)
ut_case.add_case(["Ascend910A"], case13)
ut_case.add_case(["Ascend910A"], case14)
ut_case.add_case(["Ascend910A"], case15)
ut_case.add_case(["Ascend910A"], case16)
ut_case.add_case(["Ascend910A"], case17)
ut_case.add_case(["Ascend910A"], case18)
ut_case.add_case(["Ascend910A"], case19)
ut_case.add_case(["Ascend910A"], case20)
ut_case.add_case(["Ascend910A"], case21)
ut_case.add_case(["Ascend910A"], case22)
ut_case.add_case(["Ascend910A"], case23)
ut_case.add_case(["Ascend910A"], case24)
ut_case.add_case(["Ascend910A"], case25)
ut_case.add_case(["Ascend910A"], case26)


if __name__ == '__main__':
    ut_case.run("Ascend910A")