#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ThresholdV2D", "impl.threshold_v2_d", "threshold_v2_d")

# pylint: disable=locally-disabled,too-many-arguments
def gen_threshold_v2_d_case(shape_x, dtype_x, format_var,threshold,
                            value, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "format": format_var,
                        "dtype": dtype_x, "ori_shape": shape_x,
                        "ori_format": format_var},
                       {"shape": shape_x, "format": format_var,
                        "dtype": dtype_x, "ori_shape": shape_x,
                        "ori_format": format_var},
                       threshold, value],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 6, 7, 8, 9), "float32",
                                        "ND", 4.5, 3.6,
                                        "test_float32_case0",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((4, 5), "float32",
                                        "ND", 5.6, 8.6,
                                        "test_float32_case1",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((2, 8, 10), "float16",
                                        "ND", 10.5, 8.6,
                                        "test_float16_case0",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((6, 2, 8, 10), "float16",
                                        "ND", 6.6, 7.4,
                                        "test_float16_case1",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((6, 7), "int32",
                                        "ND", 5.5, 3.1,
                                        "test_int32_case0",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 6, 7, 8, 9, 10), "int32",
                                        "ND", 7.5, 9.6,
                                        "test_int32_case1",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 6, 9), "int8",
                                        "ND", 20.0, 50.6,
                                        "test_int8_case0",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 6, 9, 15, 11), "int8",
                                        "ND", 7.0, 40.8,
                                        "test_int8_case1",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 6, 7, 8, 9), "uint8",
                                        "ND", 4.5, 3.6,
                                        "test_uint8_case0",
                                        "success"))
ut_case.add_case("all",
                gen_threshold_v2_d_case((5, 8, 9), "uint8",
                                        "ND", 10.6, 7.1,
                                        "test_uint8_case1",
                                        "success"))
if __name__ == '__main__':
    ut_case.run()
    exit(0)