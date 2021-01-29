#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("Bn3dTrainingReduceGrad", None, None)

def gen_bn_3d_training_reduce_grad_case(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                     shape_batch_variance, dtype, dtype_others, case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_x_norm, "ori_shape": shape_x_norm, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_diff_scale, "ori_shape": shape_diff_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_diff_offset, "ori_shape": shape_diff_offset, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NC1HWC0", "ori_format":"NC1HWC0"},
                       {"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NC1HWC0", "ori_format":"NC1HWC0"}],
            "case_name": case_name_val,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}
def gen_bn_3d_training_reduce_grad_case1(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                     shape_batch_variance, dtype, dtype_others, case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_x_norm, "ori_shape": shape_x_norm, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_diff_scale, "ori_shape": shape_diff_scale, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_diff_offset, "ori_shape": shape_diff_offset, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_scale, "ori_shape": shape_scale, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_batch_variance, "ori_shape": shape_batch_variance, "dtype":dtype_others, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                       {"shape":shape_grads, "ori_shape": shape_grads, "dtype":dtype, "format":"NDC1HWC0", "ori_format":"NDC1HWC0"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}
case5 = gen_bn_3d_training_reduce_grad_case1((2,3,2,2,2,16), (2,3,2,2,2,16), (1,1,2,1,1,16), (1,1,2,1,1,16), (1,1,2,1,1,16),
                                         (1,1,2,1,1,16), "float16", "float32", "bn_training_reduce_grad_5")


ut_case.add_case(["Ascend910A"], case5)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
