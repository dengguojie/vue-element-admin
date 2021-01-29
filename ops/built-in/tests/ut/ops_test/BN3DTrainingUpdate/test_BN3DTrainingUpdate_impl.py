#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("Bn3dTrainingUpdate", None, None)


def gen_bn_3d_training_update_case2(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset,
                                shape_mean, shape_variance, dtype, dtype_others, factor, epsilon, case_name_val):
    return {"params": [{"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_square_sum, "ori_shape": shape_square_sum, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       factor, epsilon],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case2 = gen_bn_3d_training_update_case2((2,1,2,5,5,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16),
                                    "float16", "float32", 0.2, 0.0001,"bn_3D_training_update_2")

ut_case.add_case(["Ascend910A"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")

