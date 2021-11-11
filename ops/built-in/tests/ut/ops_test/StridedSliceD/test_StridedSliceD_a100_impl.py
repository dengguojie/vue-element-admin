"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

StridedSliceD slice with last dim in a100 ut case
"""
import tbe
from tbe.common.platform import set_current_compile_soc_info
from op_test_frame.ut import OpUT
from impl.strided_slice_d import strided_slice_d

ut_case = OpUT("StridedSliceD", "impl.strided_slice_d", "strided_slice_d")


def test_a100_case1(test_arg):
    """
    test_a100_case1

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 85),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 85),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 83),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 83),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 83],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case2(test_arg):
    """
    test_a100_case2

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 85),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 85),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 1),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 1),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 1],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case3(test_arg):
    """
    test_a100_case3

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 2),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 2),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 1),
             "dtype": "float16", "format": "ND", "ori_shape": (80, 3, 76, 76, 1),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 1],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case4(test_arg):
    """
    test_a100_case4

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 85),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 85),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 83),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 83),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 83],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case5(test_arg):
    """
    test_a100_case5

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 85),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 85),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 1),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 1),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 1],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case6(test_arg):
    """
    test_a100_case6

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (80, 3, 76, 76, 2),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 2),
             "ori_format": "ND"},
            {"shape": (80, 3, 76, 76, 1),
             "dtype": "float32", "format": "ND", "ori_shape": (80, 3, 76, 76, 1),
             "ori_format": "ND"},
            [0, 0, 0, 0, 0],
            [80, 3, 76, 76, 1],
            [1, 1, 1, 1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case7(test_arg):
    """
    test_a100_case7

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (99, 110007),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 110007),
             "ori_format": "ND"},
            {"shape": (99, 110001),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 110001),
             "ori_format": "ND"},
            [0, 1],
            [99, 110002],
            [1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case8(test_arg):
    """
    test_a100_case8

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (99, 98301),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 98301),
             "ori_format": "ND"},
            {"shape": (99, 98299),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 98299),
             "ori_format": "ND"},
            [0, 1],
            [99, 98300],
            [1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case9(test_arg):
    """
    test_a100_case9

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (99, 55009),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 55009),
             "ori_format": "ND"},
            {"shape": (99, 55006),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 55006),
             "ori_format": "ND"},
            [0, 1],
            [99, 55007],
            [1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case10(test_arg):
    """
    test_a100_case10

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (99, 98301),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 98301),
             "ori_format": "ND"},
            {"shape": (99, 37),
             "dtype": "float16", "format": "ND", "ori_shape": (99, 37),
             "ori_format": "ND"},
            [0, 4000],
            [99, 4037],
            [1, 1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


def test_a100_case11(test_arg):
    """
    test_a100_case11

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    with tbe.common.context.op_context.OpContext("pre-static"):
        strided_slice_d(
            {"shape": (9,),
             "dtype": "float16", "format": "ND", "ori_shape": (9,),
             "ori_format": "ND"},
            {"shape": (7,),
             "dtype": "float16", "format": "ND", "ori_shape": (7,),
             "ori_format": "ND"},
            [1],
            [8],
            [1],
            0, 0, 0, 0, 0)

    set_current_compile_soc_info(test_arg)


# ut_case.add_cust_test_func(test_func=test_a100_case1)
# ut_case.add_cust_test_func(test_func=test_a100_case2)
# ut_case.add_cust_test_func(test_func=test_a100_case3)
# ut_case.add_cust_test_func(test_func=test_a100_case4)
# ut_case.add_cust_test_func(test_func=test_a100_case5)
# ut_case.add_cust_test_func(test_func=test_a100_case6)
# ut_case.add_cust_test_func(test_func=test_a100_case7)
# ut_case.add_cust_test_func(test_func=test_a100_case8)
# ut_case.add_cust_test_func(test_func=test_a100_case9)
# ut_case.add_cust_test_func(test_func=test_a100_case10)
# ut_case.add_cust_test_func(test_func=test_a100_case11)
