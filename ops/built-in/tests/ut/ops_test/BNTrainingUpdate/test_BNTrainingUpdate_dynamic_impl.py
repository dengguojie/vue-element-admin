#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("BNTrainingUpdate", "impl.dynamic.bn_training_update", "bn_training_update")

def gen_bn_training_update_case(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset,
                                shape_mean, shape_variance, dtype, dtype_others, factor, epsilon, case_name_val, expected):
    return {"params": [{"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((2, None), (2, None), (2, None), (2, None), (2, None))},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_square_sum, "ori_shape": shape_square_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_mean, "ori_shape": shape_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_variance, "ori_shape": shape_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       factor, epsilon],
            "case_name": case_name_val,
            "expect": expected,
            "format_expect": [],
            "support_expect": True}

case1 = gen_bn_training_update_case((-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), 
                                    "float16", "float32", 0.2, 0.0001,"bn_training_update_1", "success")
case2 = gen_bn_training_update_case((256,2,112,112,16), (1,2,1,1,16), (1,2,1,1,16), (1,2,1,1,16), (1,2,1,1,16), (1,2,1,1,16), (1,2,1,1,16), 
                                    "float16", "float32", 0.2, 0.0001,"bn_training_update_2", "success")
case3 = gen_bn_training_update_case((-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), (-1,-1,-1,-1,-1), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_3", "success")
case4 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_4", "success")
case5 = gen_bn_training_update_case((32,32,28,28,16), (1,31,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_5", RuntimeError)
case6 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,2,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_6", RuntimeError)
case7 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,31,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_7", RuntimeError)
case8 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,32,1,1,16), (1,2,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_8", RuntimeError)
case9 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,2,1,1,16), (1,32,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_9", RuntimeError)
case10 = gen_bn_training_update_case((32,32,28,28,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,32,1,1,16), (1,2,1,1,16), (1,32,1,1,16), 
                                    "float32", "float32", 0.2, 0.0001,"bn_training_update_10", RuntimeError)

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)

if __name__ == '__main__':
    ut_case.run("Ascend910A")

