#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("Tile", "impl.dynamic.tile", "tile")

case1 = {"params": [
        {"ori_shape": (-3,), "shape": (-3,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, None),)},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": ((1, None),)},
        {"ori_shape": (-3,), "shape": (-3,), "ori_format": "ND",
         "format": "ND", "dtype": "float16", "range": ((1, None),)}],
         "case_name": "tile_case_error_1",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

case2 = {"params": [
        {"ori_shape": (-1, -1), "shape": (-1, -1), "ori_format": "ND", "format": "ND",
         "dtype": "float16", "range": ((1, None), (1, None))},
        {"ori_shape": (1,), "shape": (1,), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": ((1, None),)},
        {"ori_shape": (-1,), "shape": (-1,), "ori_format": "ND", "format": "ND",
         "dtype": "float16", "range": ((1, None), (1, None))}],
         "case_name": "tile_case_error_2",
         "expect": RuntimeError, "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

from impl.dynamic.tile import adapt_shape_compute

def reload_check_support():
	"""
	reload_check_support to improve cov
	"""
	adapt_shape_compute([1, 1, 1, 1], [(1, 1), (1, 1), (1, 1), (1, 1)], (4,), [16, 16, 16, 16])

if __name__ == '__main__':
    reload_check_support()
    ut_case.run("Ascend910A")