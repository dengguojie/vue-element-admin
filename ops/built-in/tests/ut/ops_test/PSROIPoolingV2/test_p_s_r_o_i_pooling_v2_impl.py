#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("p_s_r_o_i_pooling_v2")

def gen_psroipooling_case(x_dict, rois_dict, y_dict, output_dim, group_size,
                          spatial_scale, kernel_name_val, expect, calc_expect_func=None):
    if calc_expect_func:
        return {"params": [x_dict, rois_dict, y_dict, output_dim, group_size, spatial_scale],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True,
                "calc_expect_func": calc_expect_func}
    else:
        return {"params": [x_dict, rois_dict, y_dict, output_dim, group_size, spatial_scale],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True}

# invalid dtype
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "int32", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "int32", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "int32", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_01", RuntimeError))

# invalid dtype 3
ut_case.add_case("Ascend910",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_03", RuntimeError))

# invalid shape 1
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_04", RuntimeError))

# invalid shape 2
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_05", RuntimeError))

# invalid shape 3
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 25), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 25), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_06", RuntimeError))

# invalid shape 4
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 3*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 3*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.0625,
                     "psroipooling_07", RuntimeError))

# invalid group_size
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 4, 0.0625,
                     "psroipooling_08", RuntimeError))

# invalid group_size 2
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*129*129, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*129*129, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 129, 129, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 129, 129, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 129, 0.0625,
                     "psroipooling_09", RuntimeError))

# invalid output_dim
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 2, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     33, 3, 0.0625,
                     "psroipooling_10", RuntimeError))

# normal shape 1
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 1*1*1, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 1*1*1, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 1, 1, 1, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 1, 0.0625,
                     "psroipooling_11", "success"))

# normal shape 2
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*7*7, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*7*7, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 2, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (6, 2, 7, 7, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 7, 0.125,
                     "psroipooling_12", "success"))

# normal shape 3
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 9*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 9*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 9, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (6, 9, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     140, 3, 0.125,
                     "psroipooling_13", "success"))

# normal shape 4
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625,
                     "psroipooling_14", "success"))

# normal shape 5, and fp32
ut_case.add_case("Ascend710",
                 gen_psroipooling_case(
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 11), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 11), "ori_format": "ND", "param_type": "input"},
                     {"shape": (11, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (11, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     13, 3, 0.125,
                     "psroipooling_15", "success"))

# normal shape 6, and fp32
ut_case.add_case("Ascend710",
                 gen_psroipooling_case(
                     {"shape": (3, 2*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (3, 2*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (3, 5, 11), "dtype": "float32", "format": "ND",
                      "ori_shape": (3, 5, 11), "ori_format": "ND", "param_type": "input"},
                     {"shape": (11*3, 2, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (11*3, 2, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 3, 0.125,
                     "psroipooling_16", "success"))

# invalid rois shape
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*7*7, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*7*7, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 6, 6), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 6, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 2, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (6, 2, 7, 7, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 7, 0.125,
                     "psroipooling_17", RuntimeError))

# invalid rois shape 2
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*7*7, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 2*7*7, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (2, 5, 6), "dtype": "float16", "format": "ND",
                      "ori_shape": (2, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 2, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (6, 2, 7, 7, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 7, 0.125,
                     "psroipooling_18", RuntimeError))

ut_case.add_case("Ascend310",
                 gen_psroipooling_case(
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 11), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 11), "ori_format": "ND", "param_type": "input"},
                     {"shape": (11, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (11, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     13, 3, 0.125,
                     "psroipooling_19", "success"))
                     
# output_dim less than or equal 1
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 1*1*1, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 1*1*1, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (15, 1, 1, 1, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     1, 1, 0.0625,
                     "psroipooling_20", "success"))

# output_dim less than or equal 1
ut_case.add_case("Ascend910",
                 gen_psroipooling_case(
                     {"shape": (1, 1*1*1, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*1*1, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (15, 1, 1, 1, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     1, 1, 0.0625,
                     "psroipooling_21", "success"))


# normal shape 1
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 1*1*1, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*1*1, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 15), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 15), "ori_format": "ND", "param_type": "input"},
                     {"shape": (15, 1, 1, 1, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (15, 1, 1, 1, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 1, 0.0625,
                     "psroipooling_22", "success"))

# normal shape 2
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 2*7*7, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 2*7*7, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (6, 2, 7, 7, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     21, 7, 0.125,
                     "psroipooling_23", "success"))

# normal shape 3
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 9*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 9*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (6, 9, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (6, 9, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     140, 3, 0.125,
                     "psroipooling_24", "success"))

# normal shape 4
ut_case.add_case("all",
                 gen_psroipooling_case(
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625,
                     "psroipooling_25", "success"))
              

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend910"])
    exit(0)
