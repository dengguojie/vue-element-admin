#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Tile", "impl.dynamic.tile", "tile")


def gen_dynamic_tile_d_case(shape_x, range_x, multiples, dtype_val,
                            kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": (len(multiples),), "shape": (len(multiples),), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": range_x},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}

def gen_dynamic_tile_case(shape_x, range_x, multiples, dtype_val,
                            kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": (-1,), "shape": (-1,), "ori_format": "ND",
         "format": "ND", "dtype": "int32", "range": range_x},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}

ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [2],
                                                "float16",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))
ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [2],
                                                "int8",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))
ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [2],
                                                "bool",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))

ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [2,2],
                                                "int8",
                                                "dynamic_tile_0",
                                                "success"))

ut_case.add_case("all", gen_dynamic_tile_d_case((-1,1), ((1, None),(1, None)), [2,2],
                                                "int8",
                                                "dynamic_tile_1",
                                                "success"))

ut_case.add_case("all", gen_dynamic_tile_d_case((-1,2), ((1, None),(1, None)), [2,2],
                                                "int8",
                                                "dynamic_tile_2",
                                                "success"))

ut_case.add_case("all", gen_dynamic_tile_case((-1,), ((1, None),), [2],
                                                "float16",
                                                "dynamic_tile_3",
                                                "success"))

ut_case.add_case("all", gen_dynamic_tile_d_case((-3,), ((1, None),), [2],
                                                "float16",
                                                "dynamic_tile_error_1",
                                                RuntimeError))

ut_case.add_case("all", gen_dynamic_tile_d_case((-1, -1), ((1, None),(1, None)), [2],
                                                "float16",
                                                "dynamic_tile_error_2",
                                                RuntimeError))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
