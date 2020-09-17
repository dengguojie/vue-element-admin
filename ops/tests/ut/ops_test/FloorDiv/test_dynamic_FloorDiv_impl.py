#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("FloorDiv", "impl.dynamic.floor_div", "floor_div")


def gen_dynamic_floor_div_case(shape_x, shape_y, range_x, range_y, dtype_val,
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


ut_case.add_case("all", gen_dynamic_floor_div_case((16,), (1,), ((16, 16),),
                                                   ((1, 1),), "float16",
                                                   "dynamic_floor_div_fp16_ND",
                                                   "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
