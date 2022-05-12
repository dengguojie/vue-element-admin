#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import json
ut_case = OpUT("BNTrainingUpdate", "impl.bn_training_update", "bn_training_update")

def gen_bn_training_update_case(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset,
                                shape_mean, shape_variance, dtype, dtype_others, factor, epsilon, case_name_val):
    return {"params": [{"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_square_sum, "ori_shape": shape_square_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       factor, epsilon],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}
def gen_bn_training_update_case2(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset,
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
case1 = gen_bn_training_update_case((2,4,384,576,16), (1,4,1,1,16), (1,4,1,1,16), (1,4,1,1,16), (1,4,1,1,16), (1,4,1,1,16), (1,4,1,1,16),
                                    "float16", "float32", 0.2, 0.0001,"bn_training_update_1")
case2 = gen_bn_training_update_case2((2,1,2,5,5,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16),
                                    "float16", "float32", 0.2, 0.0001,"bn_training_update_2")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

def test_get_op_support_info(test_arg):
    from impl.bn_training_update import get_op_support_info
    res = get_op_support_info({"shape":(2,1,2,5,5,16), "ori_shape": (2,1,2,5,5,16), "dtype":"float16", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(2,1,2,5,5,16), "ori_shape": (2,1,2,5,5,16), "dtype":"float16", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        {"shape":(1,4,1,1,16), "ori_shape": (1,4,1,1,16), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                        0.2, 0.0001)
    split_maps = json.loads(res).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 1
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 1
        idx = input_list[0].get("idx")
        assert idx == 0
        axis = input_list[0].get("axis")
        assert axis == [0]
        headOverLap = input_list[0].get("headOverLap")
        assert headOverLap == [-1]
        tailOverLap = input_list[0].get("tailOverLap")
        assert tailOverLap == [-1]
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

if __name__ == '__main__':
    ut_case.run("Ascend910A")

