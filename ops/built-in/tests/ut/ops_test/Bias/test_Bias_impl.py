#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.util import util_common

ut_case = OpUT("Bias", None, None)

case1 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (1, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 4), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 4),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (3, 3, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float32", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (3, 3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3, 3),"ori_format": "NCHW"},
                    1, 1],
         "case_name": "bias_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    3],
         "case_name": "bias_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, -2],
         "case_name": "bias_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3,), "dtype": "float16", "format": "NCHW", "ori_shape": (3,),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    -1, -1],
         "case_name": "bias_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    2, -1],
         "case_name": "bias_8",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    2, -1, False],
         "case_name": "bias_9",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3),"ori_format": "NCHW"},
                    {"shape": (3, 3, 3), "dtype": "float16", "format": "NCHW", "ori_shape": (3, 3, 3),"ori_format": "NCHW"},
                    1, 1, False],
         "case_name": "bias_10",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case1)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case2)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case3)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case4)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case5)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case6)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case7)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case8)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case9)
ut_case.add_case(["Ascend910", "Hi3796CV300CS"], case10)

def test_op_select_format(test_arg):
    """
    test_op_select_format
    """
    from impl.bias import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     "test_bias_op_select_format_1")
    op_select_format({"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     {"shape": (-1, 1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 1), "ori_format": "ND"},
                     "test_bias_op_select_format_2")
    op_select_format({"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 1, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 1, 1, 16),
                      "ori_format": "NHWC"},
                     "test_bias_op_select_format_3")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     "test_bias_op_select_format_4")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_5")
    op_select_format({"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,), "ori_format": "NCHW"},
                     "test_bias_op_select_format_6")
    op_select_format({"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_7")
    op_select_format({"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 16, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 16, 16),
                      "ori_format": "NCHW"},
                     {"shape": (16, 16, 32, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (16, 16, 32, 16),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_8")
    op_select_format({"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,), "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "float16", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_9")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "float16", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_bias_op_select_format_10")
    op_select_format({"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (1, 16, 1, 1), "dtype": "uint8", "format": "NCHW", "ori_shape": (1, 16, 1, 1),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_11")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     {"shape": (1, 1, 32, 1), "dtype": "int8", "format": "HWCN", "ori_shape": (1, 1, 32, 1),
                      "ori_format": "HWCN"},
                     "test_bias_op_select_format_12")
    op_select_format({"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     {"shape": (1,), "dtype": "int8", "format": "NHWC", "ori_shape": (1,),
                      "ori_format": "NHWC"},
                     {"shape": (1, 1, 16, 1), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 1, 16, 1),
                      "ori_format": "NHWC"},
                     "test_bias_op_select_format_13")
    op_select_format({"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     {"shape": (1,64,1,1), "dtype": "float16", "format": "NCHW", "ori_shape": (1,64,1,1),
                      "ori_format": "NCHW"},
                     {"shape": (4,64,200,320), "dtype": "float16", "format": "NCHW", "ori_shape": (4,64,200,320),
                      "ori_format": "NCHW"},
                     "test_bias_op_select_format_14")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     "test_bias_op_select_format_15")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN"},
                     "test_bias_op_select_format_16")
    op_select_format({"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 1},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     {"shape": (3, 3, 16, 128), "dtype": "float32", "format": "HWCN", "ori_shape": (3, 3, 16, 128),
                      "ori_format": "HWCN", "sub_format" : 8},
                     "test_bias_op_select_format_17")
    op_select_format({"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (),
                      "ori_format": "NHWC", "sub_format" : 0},
                     "test_bias_op_select_format_18")

    def __test_util_commom():
        input_parm = ({"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0},
                      {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0},
                      {"shape": (1,), "dtype": "float32", "format": "NHWC", "ori_shape": (1,), "ori_format": "NHWC", "sub_format" : 0})
        util_common.is_support_fractal_z_inputs(input_parm)

    __test_util_commom()

ut_case.add_cust_test_func(test_func=test_op_select_format)

ut_case.run(["Ascend910", "Ascend310", "Ascend710", "Hi3796CV300CS"])

