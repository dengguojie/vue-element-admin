#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Tile", "impl.dynamic.select", "select")

case1 = {"params": [
        {"ori_shape": (-1, -1), "shape": (-1, -1), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, None),)},
        {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": ((1, None),)},
        {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": ((1, None),)},
        {"ori_shape": (-1, -1, -1), "shape": (-1, -1, -1), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, None),)}],
         "case_name": "tile_case_error_1",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")