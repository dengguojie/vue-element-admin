#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.arg_max_with_kd import op_select_format
from tbe.common.platform import set_current_compile_soc_info

ut_case = OpUT("ArgMaxWithKD", None, None)


def test_op_select_format_000(test_arg):
    """
    test_op_select_format_000
    """
    op_select_format(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        0,
        False,
        1,
    )


def test_op_select_format_001(test_arg):
    """
    test_op_select_format_001
    """
    op_select_format(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        10000,
        False,
        1,
    )
ut_case.add_cust_test_func(test_func=test_op_select_format_000)
ut_case.add_cust_test_func(test_func=test_op_select_format_001)

def test_op_select_format_002(test_arg):
    from te.platform.cce_conf import te_set_version
    te_set_version("Ascend710")
    op_select_format(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NCHW"
        },
        10000,
        False,
        1,
    )

    op_select_format(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        0,
        False,
        1,
    )

    op_select_format(
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        {
            "shape": (5, 8, 16, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (5, 8, 16, 16),
            "ori_format": "NHWC"
        },
        10000,
        False,
        1,
    )
    te_set_version("Ascend710")
ut_case.add_cust_test_func(test_func=test_op_select_format_002)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])
