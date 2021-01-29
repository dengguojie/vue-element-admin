#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("Bn3dTrainingUpdateGrad", None, None)


def gen_bn_3d_training_update_grad_case1(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, dtype, dtype_others,
                                     case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_mean, "ori_shape": shape_batch_mean, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

case5 = gen_bn_3d_training_update_grad_case1((2,1,2,5,5,16), (2,1,2,5,5,16), (1,1,2,1,1,16), (1,1,2,1,1,16), "float16", "float32",
                                         "bn_training_update_grad_5")

ut_case.add_case(["Ascend910A"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
