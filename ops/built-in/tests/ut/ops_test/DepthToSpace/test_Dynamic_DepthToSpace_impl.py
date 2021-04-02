#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# pylint: disable=invalid-name,missing-docstring
from op_test_frame.ut import OpUT

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


ut_case.add_case(
    "all",
    gen_dynamic_depthtospace_case((-1, -1, -1, -1), (-1, -1, -1, -1), (-1, -1, -1, -1), (-1, -1, -1, -1),
                                  ((1, None), (1, None), (1, None), (1, None)),
                                  ((1, None), (1, None), (1, None), (1, None)), "NHWC", "NHWC", "float16",
                                  "depthtospace_case", 2, "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
