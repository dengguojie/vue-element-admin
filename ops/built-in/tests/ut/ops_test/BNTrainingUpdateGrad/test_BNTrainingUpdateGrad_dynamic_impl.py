#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT 
import tbe
ut_case = OpUT("BNTrainingUpdateGrad", "impl.dynamic.bn_training_update_grad", "bn_training_update_grad")

def gen_bn_training_update_grad_case(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, dtype, dtype_others,
                                     case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]},
                       {"shape":shape_batch_mean, "ori_shape": shape_batch_mean, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0", "range": [(1,None), (1,None), (1,None), (1,None), (1,None)]}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case1 = gen_bn_training_update_grad_case((-1,-1,-1,-1,16), (-1,-1,-1,-1,16), (1,-1,1,1,16), (1,-1,1,1,16), "float32", "float32",
                                         "bn_training_update_grad_1")

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
