#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingUpdateGrad", None, None)

def gen_bn_training_update_grad_case(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, dtype, dtype_others,
                                     case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_mean, "ori_shape": shape_batch_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"}],
            "case_name": case_name_val,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

case1 = gen_bn_training_update_grad_case((2,2,2,2,16), (2,2,2,2,16), (1,3,1,1,16), (1,2,1,1,16), "float16", "float32",
                                         "bn_training_update_grad_1")
case2 = gen_bn_training_update_grad_case((2,2,2,2,16), (2,2,2,2,16), (1,2,2,1,16), (1,2,1,1,16), "float16", "float32",
                                         "bn_training_update_grad_2")
case3 = gen_bn_training_update_grad_case((2,3,4,2,19,16), (2,3,4,2,16), (1,3,1,1,16), (1,3,1,1,16), "float16", "float32",
                                         "bn_training_update_grad_3")
case4 = gen_bn_training_update_grad_case((2,3,4,2,16), (2,3,4,2,16), (1,3,1,1,16), (1,3,1,1,17), "float16", "float32",
                                         "bn_training_update_grad_4")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
