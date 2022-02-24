#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BNTrainingUpdateV2", "impl.dynamic.bn_training_update_v2", "bn_training_update_v2")

def gen_bn_training_update_v2_case(shape_x, shape_sum, shape_square_sum, shape_scale, shape_offset,
                                   dtype, dtype_others, epsilon, case_name_val):
    return {"params": [{"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_square_sum, "ori_shape": shape_square_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_offset, "ori_shape": shape_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       {"shape":shape_sum, "ori_shape": shape_sum, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": ((1, None), (1, None), (1, None), (1, None), (1, None))},
                       epsilon],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_bn_training_update_v2_case((-1,-1,-1,-1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), 
                                    "float16", "float32", 0.0001,"bn_training_update_v2_1")
case2 = gen_bn_training_update_v2_case((-1,-1,-1,-1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), (1,-1,1,1,-1), 
                                    "float32", "float32", 0.0001,"bn_training_update_v2_2")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)

#if __name__ == '__main__':
#    ut_case.run("Ascend910A")

