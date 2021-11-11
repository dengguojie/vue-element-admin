#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
from impl.softmax_v2 import get_op_support_info, \
                            compute_nz_nopad_fp32, compute_nopad, compute_nz_nopad, \
                            compute_nz_padding, compute_padding, compute_padding_fp32
from te import tvm
ut_case = OpUT("SoftmaxV2", None, None)

case1 = {"params": [{"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"},
                    {"shape": (1,2,4), "dtype": "float16", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND"}],
         "case_name": "softmax_v2_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    {"shape": (16,16), "dtype": "float16", "format": "ND", "ori_shape": (16,16),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float16", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    {"shape": (32, 2, 4, 16), "dtype": "float32", "format": "ND", "ori_shape": (32, 2, 4, 16),"ori_format": "ND"},
                    [1,3]],
         "case_name": "softmax_v2_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND"},
                    [0,1]],
         "case_name": "softmax_v2_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_6hd_n",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_6hd_c",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [0]],
         "case_name": "softmax_v2_6hd_n",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (16, 16, 1, 16, 16, 16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    {"shape": (16, 16, 1, 16, 16 ,16), "dtype": "float32", "format": "NDC1HWC0", "ori_shape": (16, 16, 16, 16, 16),"ori_format": "NDHWC"},
                    [-1]],
         "case_name": "softmax_v2_6hd_c",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (16, 16, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 16, 16),"ori_format": "ND"},
                     {"shape": (16, 16, 1, 1, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 16, 16),"ori_format": "ND"},
                     [-1]],
         "case_name": "softmax_v2_nz",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (16, 16, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                     {"shape": (16, 16, 4, 4, 16 ,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (16, 16, 50, 50),"ori_format": "ND"},
                     [-1]],
         "case_name": "softmax_v2_nz_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [{"shape": (16, 1, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     {"shape": (16, 1, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     [-1]],
         "case_name": "softmax_v2_5hd_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [{"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                     {"shape": (8, 6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 8732, 81),"ori_format": "ND"},
                     [2]],
          "case_name": "softmax_v2_nz_13",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}

case14 = {"params": [{"shape": (6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8732, 81),"ori_format": "ND"},
                    {"shape": (6, 546, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8732, 81),"ori_format": "ND"},
                    [-1]],
         "case_name": "softmax_v2_nz_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case15 = {"params": [{"shape": (16, 1, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     {"shape": (16, 1, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16, 4, 4, 4),"ori_format": "NHWC"},
                     [-1]],
         "case_name": "softmax_v2_5hd_02",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend710","Ascend910A","Hi3796CV300CS"], case14)
ut_case.add_case(["Ascend310", "Ascend910A"], case15)
# precision cases
## need axis is list

def calc_expect_func(x, y, axis):
    input_Arr = x['value']
    data_max = np.max(input_Arr, axis, keepdims=True).astype(np.float16)

    data_sub = np.subtract(input_Arr, data_max).astype(np.float32)
    expres = np.exp(data_sub).astype(np.float32)
    sumre = np.sum(expres, axis, keepdims=True).astype(np.float32)
    result = (expres / sumre).astype(y['dtype'])
    return result

ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1, 2), "dtype": "float32", "format": "ND", "ori_shape": (1, 2),"ori_format": "ND", "param_type": "output"},
                                                    1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (16,16), "dtype": "float32", "format": "ND", "ori_shape": (16,16),"ori_format": "ND", "param_type": "output"},
                                                    0],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (32,2,4,16), "dtype": "float32", "format": "ND", "ori_shape": (32,2,4,16),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (32,2,4,16), "dtype": "float32", "format": "ND", "ori_shape": (32,2,4,16),"ori_format": "ND", "param_type": "output"},
                                                    1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })
ut_case.add_precision_case("Ascend910", {"params": [{"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "input"},
                                                    {"shape": (1,2,4), "dtype": "float32", "format": "ND", "ori_shape": (1,2,4),"ori_format": "ND", "param_type": "output"},
                                                    -1],
                                         "calc_expect_func": calc_expect_func,
                                         "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                         })


def test_get_op_support_info_000(test_arg):
    """
    test_get_op_support_info_000
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        -4,
    )


def test_get_op_support_info_001(test_arg):
    """
    test_get_op_support_info_001
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        -3,
    )


def test_get_op_support_info_002(test_arg):
    """
    test_get_op_support_info_002
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        -2,
    )


def test_get_op_support_info_003(test_arg):
    """
    test_get_op_support_info_003
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NCHW"
        },
        -1,
    )


def test_get_op_support_info_004(test_arg):
    """
    test_get_op_support_info_004
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -4,
    )


def test_get_op_support_info_005(test_arg):
    """
    test_get_op_support_info_005
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -3,
    )


def test_get_op_support_info_006(test_arg):
    """
    test_get_op_support_info_006
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -2,
    )


def test_get_op_support_info_007(test_arg):
    """
    test_get_op_support_info_007
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NC1HWC0",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -1,
    )


def test_get_op_support_info_008(test_arg):
    """
    test_get_op_support_info_008
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "FRACTAL_NZ",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "FRACTAL_NZ",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -1,
    )


def test_get_op_support_info_009(test_arg):
    """
    test_get_op_support_info_009
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -1,
    )


def test_get_op_support_info_010(test_arg):
    """
    test_get_op_support_info_010
    """
    get_op_support_info(
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NE",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        {
            "shape": (16, 1, 4, 4, 16),
            "dtype": "float32",
            "format": "NE",
            "ori_shape": (16, 4, 4, 4),
            "ori_format": "NHWC"
        },
        -1,
    )


def test_compute_nz_nopad_fp32_001(test_arg):
    """
    test_compute_nz_nopad_fp32_001
    """
    compute_nz_nopad_fp32(
        tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="float32"),
        [32, 2, 4, 16],
    )


def test_compute_nz_nopad_fp32_002(test_arg):
    """
    test_compute_nz_nopad_fp32_002
    """
    compute_nz_nopad_fp32(
        tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="float16"),
        [32, 2, 4, 16],
    )


def test_compute_nz_nopad_fp32_003(test_arg):
    """
    test_compute_nz_nopad_fp32_003
    """
    compute_nz_nopad_fp32(
        tvm.placeholder((32, 2, 4, 16), name='tensor_in', dtype="int16"),
        [32, 2, 4, 16],
    )


def test_compute_nopad_001(test_arg):
    """
    test_compute_nopad_001
    """
    compute_nopad(
        tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
        [32, 2, 4, 16, 4],
    )


def test_compute_nz_nopad_001(test_arg):
    """
    test_compute_nz_nopad_001
    """
    compute_nz_nopad(
        tvm.placeholder((32, 2, 2, 16), name='tensor_in', dtype="float16"),
        [32, 2, 2, 16],
    )


def test_compute_nz_padding_001(test_arg):
    """
    test_compute_nz_padding_001
    """
    compute_nz_padding(
        tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float16"),
        [6, 546, 16, 16],
        [6, 15],
    )


def test_compute_nz_padding_002(test_arg):
    """
    test_compute_nz_padding_002
    """
    compute_nz_padding(
        tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float16"),
        [6, 546, 16, 16],
        [1, 20],
    )


def test_compute_padding_001(test_arg):
    """
    test_compute_padding_001
    """
    compute_padding(
        tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
        [32, 2, 4, 16, 4],
        [6, 15],
    )


def test_compute_padding_002(test_arg):
    """
    test_compute_padding_002
    """
    compute_padding(
        tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
        [32, 2, 4, 16, 4],
        [1, 20],
    )


def test_compute_padding_fp32_001(test_arg):
    """
    test_compute_padding_fp32_001
    """
    compute_padding_fp32(
        tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float16"),
        [32, 2, 4, 16, 4],
        [6, 15],
        "high_performance",
    )


def test_compute_padding_fp32_002(test_arg):
    """
    test_compute_padding_fp32_002
    """
    compute_padding_fp32(
        tvm.placeholder((32, 2, 4, 16, 4), name='tensor_in', dtype="float32"),
        [32, 2, 4, 16, 4],
        [1, 20],
        "high_performance",
    )

def test_multicore_factor_calculate(test_arg):
    from impl.softmax_v2 import multicore_factor_calculate
    multicore_factor_calculate([32, 2, 4, 4, 16])
    multicore_factor_calculate([1, 1, 32, 4, 16])
    multicore_factor_calculate([1, 1, 1, 32, 16])
    multicore_factor_calculate([1, 1, 1, 1, 32])

def test_multicore_factor_calculate_nz(test_arg):
    from impl.softmax_v2 import multicore_factor_calculate_nz
    multicore_factor_calculate_nz([1, 32, 4, 4, 16])
    multicore_factor_calculate_nz([1, 1, 32, 4, 16])
    multicore_factor_calculate_nz([1, 1, 1, 32, 16])

def test_tiling_factor_calculate(test_arg):
    from impl.softmax_v2 import tiling_factor_calculate
    tiling_factor_calculate([16, 1, 4, 4, 16], 1, 8, True)
    tiling_factor_calculate([16, 1, 4, 4, 16], 1, 64, True)
    tiling_factor_calculate([16, 1, 1024, 4, 16], 1, 64, True)
    tiling_factor_calculate([16, 1, 1024, 1025, 16], 1, 64, True)
    tiling_factor_calculate([16, 1, 4, 4, 16], 2, 8, True)
    tiling_factor_calculate([16, 1, 1024, 4, 16], 2, 1024, True)
    tiling_factor_calculate([16, 1, 1024, 1025, 16], 2, 1024, True)
    tiling_factor_calculate([16, 1, 4, 4, 16], 3, 8, True)
    tiling_factor_calculate([16, 1, 4, 4, 16], 3, 1025, True)
    tiling_factor_calculate([16, 1, 4, 4, 16], 4, 8, True)

def test_tiling_factor_calculate_nz(test_arg):
    from impl.softmax_v2 import tiling_factor_calculate_nz
    tiling_factor_calculate_nz([16, 1, 4, 4, 16], 1, 1, True)
    tiling_factor_calculate_nz([16, 1, 4, 4, 16], 1, 2, True)
    tiling_factor_calculate_nz([16, 1, 4, 8, 16], 1, 1024, True)
    tiling_factor_calculate_nz([16, 1, 4, 4, 16], 2, 8, True)
    tiling_factor_calculate_nz([16, 1, 4, 4, 16], 2, 2048, True)
    tiling_factor_calculate_nz([16, 1, 4, 4, 16], 3, 8, True)

def test_compute_nz_padding_fp32(test_arg):
    from impl.softmax_v2 import compute_nz_padding_fp32
    tensor_in = tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float16")
    compute_nz_padding_fp32(tensor_in,(6, 546, 16, 16), [6, 15])
    compute_nz_padding_fp32(tensor_in,(6, 546, 16, 16), [6, 1])

    tensor_in = tvm.placeholder((6, 546, 16, 16), name='tensor_in', dtype="float32")
    compute_nz_padding_fp32(tensor_in,(6, 6, 16, 16), [6, 15])
    compute_nz_padding_fp32(tensor_in,(6, 546, 16, 16), [6, 1])
    compute_nz_padding_fp32(tensor_in,(6, 546, 16, 16), [1, 1])

ut_case.add_cust_test_func(test_func=test_get_op_support_info_000)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_001)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_002)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_003)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_004)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_005)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_006)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_007)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_008)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_009)
ut_case.add_cust_test_func(test_func=test_get_op_support_info_010)
ut_case.add_cust_test_func(test_func=test_compute_nz_nopad_fp32_001)
ut_case.add_cust_test_func(test_func=test_compute_nz_nopad_fp32_002)
ut_case.add_cust_test_func(test_func=test_compute_nz_nopad_fp32_003)
ut_case.add_cust_test_func(test_func=test_compute_nopad_001)
ut_case.add_cust_test_func(test_func=test_compute_nz_nopad_001)
ut_case.add_cust_test_func(test_func=test_compute_nz_padding_001)
ut_case.add_cust_test_func(test_func=test_compute_nz_padding_002)
ut_case.add_cust_test_func(test_func=test_compute_padding_001)
ut_case.add_cust_test_func(test_func=test_compute_padding_002)
ut_case.add_cust_test_func(test_func=test_compute_padding_fp32_001)
ut_case.add_cust_test_func(test_func=test_compute_padding_fp32_002)
ut_case.add_cust_test_func(test_func=test_multicore_factor_calculate)
ut_case.add_cust_test_func(test_func=test_multicore_factor_calculate_nz)
ut_case.add_cust_test_func(test_func=test_tiling_factor_calculate)
ut_case.add_cust_test_func(test_func=test_tiling_factor_calculate_nz)
ut_case.add_cust_test_func(test_func=test_compute_nz_padding_fp32)

if __name__ == '__main__':
    ut_case.run(["Ascend910A","Hi3796CV300CS"])
