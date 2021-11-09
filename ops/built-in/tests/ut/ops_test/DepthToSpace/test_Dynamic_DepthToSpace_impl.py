#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name,missing-docstring
from op_test_frame.ut import OpUT
from impl.dynamic.depth_to_space import get_op_support_info

ut_case = OpUT("DepthToSpace", "impl.dynamic.depth_to_space", "depth_to_space")


def gen_dynamic_depthtospace_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, in_format, ori_format,
                                  dtype_val, kernel_name_val, block_size, expect):
    return {
        "params": [
            {
                "shape": shape_x,
                "dtype": dtype_val,
                "ori_shape": ori_shape_x,
                "ori_format": ori_format,
                "format": in_format,
                "range": range_x
            },
            {
                "shape": shape_y,
                "dtype": dtype_val,
                "ori_shape": ori_shape_y,
                "ori_format": ori_format,
                "format": in_format,
                "range": range_y
            },
            block_size,
        ],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }

case_mode_fail = {
        "params": [
            {
                "shape": (-1, -1, -1, -1),
                "dtype": "float16",
                "ori_shape": (-1, -1, -1, -1),
                "ori_format": "NHWC",
                "format": "NHWC",
                "range": ((1, None), (1, None), (1, None), (1, None))
            },
            {
                "shape": (-1, -1, -1, -1),
                "dtype": "float16",
                "ori_shape": (-1, -1, -1, -1),
                "ori_format": "NHWC",
                "format": "NHWC",
                "range": ((1, None), (1, None), (1, None), (1, None))
            },
            2,
            "ABC"
        ],
        "case_name": "depth_to_space_case_2",
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True
    }


def test_get_op_support_info_dynamic_depthtospace(test_arg):
    x = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, -1), "ori_shape": (-1, -1, -1),
         "range": ((1, 6), (16, 48), (16, 48))}
    y = {"format": "ND","ori_format": "ND", "dtype": "float16", "shape": (-1, -1, -1), "ori_shape": (-1, -1, -1),
         "range": ((16, 48), (16, 48), (16, 16))}
    block_size = 3
    get_op_support_info(x, y, block_size)


ut_case.add_cust_test_func(test_func=test_get_op_support_info_dynamic_depthtospace)
ut_case.add_case(
    "all",
    gen_dynamic_depthtospace_case((-1, -1, -1, -1), (-1, -1, -1, -1), (-1, -1, -1, -1), (-1, -1, -1, -1),
                                  ((1, None), (1, None), (1, None), (1, None)),
                                  ((1, None), (1, None), (1, None), (1, None)), "NHWC", "NHWC", "float16",
                                  "depthtospace_case", 2, "success"))
ut_case.add_case("all", case_mode_fail)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
