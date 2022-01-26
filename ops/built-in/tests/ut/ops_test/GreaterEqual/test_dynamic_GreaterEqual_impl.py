#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("GreaterEqual", "impl.dynamic.greater_equal", "greater_equal")


def gen_dynamic_greater_equal_case(shape_x, shape_y, range_x, range_y, dtype_val,
                                   kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_y},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_y}],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}


ut_case.add_case("all",
                 gen_dynamic_greater_equal_case((1,), (-1,), ((1, 1),), ((2, 16),),
                                                "float16", "dynamic_greater_equal_fp16_ND",
                                                "success"))

ut_case.add_case("all",
                 gen_dynamic_greater_equal_case((-2,), (-2,), ((1, 1),), ((2, 16),),
                                                "float16", "dynamic_greater_equal_rank_ND",
                                                "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
