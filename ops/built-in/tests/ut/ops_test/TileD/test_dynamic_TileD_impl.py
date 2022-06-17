#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TileD", "impl.dynamic.tile_d", "tile_d")


def gen_dynamic_tile_d_case(shape_x, range_x, multiples, dtype_val,
                            kernel_name_val, expect):
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
         "format": "ND", "dtype": dtype_val, "range": range_x}, multiples],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}


ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [1, 2],
                                                "float16",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))
ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [1, 2],
                                                "bool",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))
ut_case.add_case("all", gen_dynamic_tile_d_case((-1,), ((1, None),), [1, 2],
                                                "uint8",
                                                "dynamic_tile_d_fp16_ND",
                                                "success"))


def test_tile_get_op_support_info(test_arg):
    from impl.dynamic.tile_d import get_op_support_info
    get_op_support_info(
        {
            "shape": (8, 5, 5, 256),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 16, 5, 5, 16),
            "ori_format": "NC1HWC0"
        }, [1,1,1,1,1])
    get_op_support_info(
        {
            "shape": (8, 256, 5, 5),
            "dtype": "float16",
            "format": "NCHW",
            "ori_shape": (8, 256, 5, 5),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 16, 5, 5, 16),
            "ori_format": "NC1HWC0"
        }, [1,1,1,1,1])
    get_op_support_info(
        {
            "shape": (1, 16, 16),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1, 16, 16),
            "ori_format": "ND"
        }, {
            "shape": (1, 16, 16),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1, 16, 16),
            "ori_format": "ND"
        }, [1,1,1,1])


ut_case.add_cust_test_func(test_func=test_tile_get_op_support_info)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
