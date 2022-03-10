#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("p_s_r_o_i_pooling_grad_v2_d")


def gen_psroipooling_v2_d_case(x_dict, rois_dict, y_dict, output_dim, group_size,
                               spatial_scale, input_size, kernel_name_val, expect, calc_expect_func=None):
    if calc_expect_func:
        return {"params": [x_dict, rois_dict, y_dict, output_dim, group_size, spatial_scale, input_size],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True,
                "calc_expect_func": calc_expect_func}
    else:
        return {"params": [x_dict, rois_dict, y_dict, output_dim, group_size, spatial_scale, input_size],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True}


# invalid dtype
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "int32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (4, 5, 128), "dtype": "int32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "int32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 3, 0.0625, (84, 84),
                     "psroipooling_v2_d_o1", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (4, 5, 128), "dtype": "float16", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 3, 0.0625, (84, 84),
                     "psroipooling_v2_d_o2", RuntimeError))


ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (1, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 3, 0.0625, (84, 84),
                     "psroipooling_v2_d_o3", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (4, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     8, 3, 0.0625, (84, 84),
                     "psroipooling_v2_d_o4", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (4, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 3, 0.0625, (84, 84),
                     "psroipooling_v2_d_o5", RuntimeError))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (512, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (4, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (4, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (4, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 7, 0.0625, (84, 84),
                     "psroipooling_v2_d_o6", "success"))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (128, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (1, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 7, 0.0625, (84, 84),
                     "psroipooling_v2_d_o7", "success"))

# normal shape 3
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (6, 9, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (6, 9, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 9*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 9*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     140, 3, 0.125, (14, 14),
                     "psroipooling_v2_d_o8", "success"))

# normal shape 4
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float16", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (14, 14),
                     "psroipooling_v2_d_o9", RuntimeError))
# normal shape 5
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (14, 14),
                     "psroipooling_v2_d_o10", "success"))

ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (11, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (11, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 11), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 11), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     13, 3, 0.125, (14, 14),
                     "psroipooling_v2_d_o11", "success"))

# invalid input shape
# roi_shape[0] not equal y_shape[0]
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (2, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (2, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (14, 14),
                     "psroipooling_v2_d_o12", RuntimeError))

# roi_shape[1] not equal 5
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 3, 8), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 3, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (14, 14),
                     "psroipooling_v2_d_o13", RuntimeError))

# roi_shape[0] * roi_shape[2] not equal x_shape[0]
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 4), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 4), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (14, 14),
                     "psroipooling_v2_d_o14", RuntimeError))

# group size large than 128
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (6, 9, 129, 129, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (6, 9, 129, 129, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 6), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 6), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 9*129*129, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 9*129*129, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     140, 129, 0.125, (14, 14),
                     "psroipooling_v2_d_o15", RuntimeError))

# group_size not equal x_shape[2] or group_size not equal x_shape[3]
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (128, 2, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (512, 22, 7, 7), "ori_format": "NCHW", "param_type": "input"},
                     {"shape": (1, 5, 128), "dtype": "float32", "format": "ND",
                      "ori_shape": (4, 5, 128), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 68, 84, 84, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1078, 84, 84), "ori_format": "NCHW", "param_type": "output"},
                     22, 9, 0.0625, (84, 84),
                     "psroipooling_v2_d_o16", RuntimeError))

# input_size[0] not equal y_shape[2] or input_size[1] not equal y_shape[3]:
ut_case.add_case("Ascend910A",
                 gen_psroipooling_v2_d_case(
                     {"shape": (8, 1, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (8, 1, 3, 3, 16), "ori_format": "NC1HWC0", "param_type": "input"},
                     {"shape": (1, 5, 8), "dtype": "float32", "format": "ND",
                      "ori_shape": (1, 5, 8), "ori_format": "ND", "param_type": "input"},
                     {"shape": (1, 1*3*3, 14, 14, 16), "dtype": "float32", "format": "NC1HWC0",
                      "ori_shape": (1, 1*3*3, 14, 14, 16), "ori_format": "NC1HWC0", "param_type": "output"},
                     11, 3, 0.0625, (15, 15),
                     "psroipooling_v2_d_o17", RuntimeError))

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
    exit(0)
