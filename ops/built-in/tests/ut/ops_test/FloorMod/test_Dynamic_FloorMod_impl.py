#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("FloorMod", "impl.dynamic.floor_mod", "floor_mod")


def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y, dtype_val,
                              kernel_name_val, impl_mode, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_y},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_y}],
        'addition_params': {'impl_mode': impl_mode},
        "case_name": kernel_name_val, "expect": expect,
        "format_expect": [], "support_expect": True}


ut_case.add_case("all", gen_dynamic_floormod_case((16,), (1,), ((16, 16),),
                                                  ((1, 1),), "float16",
                                                  "dynamic_floormod_fp16_ND_01",
                                                  "high_performance",
                                                  "success"))

ut_case.add_case("all", gen_dynamic_floormod_case((16,), (1,), ((16, 16),),
                                                  ((1, 1),), "int32",
                                                  "dynamic_floormod_fp16_ND_02",
                                                  "high_precision",
                                                  "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
