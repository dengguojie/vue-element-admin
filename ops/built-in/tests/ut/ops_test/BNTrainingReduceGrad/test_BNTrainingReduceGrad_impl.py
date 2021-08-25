#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("BnTrainingReduceGrad", None, None)


def test_get_op_support_info(test_arg):
    from impl.bn_training_reduce_grad import get_op_support_info
    get_op_support_info({"shape": (2, 32, 2, 2), "ori_shape": (2, 32, 2, 2), "dtype": "float16", "format": "NCHW",
                         "ori_format": "NCHW"},
                        {"shape": (2, 32, 2, 2), "ori_shape": (2, 32, 2, 2), "dtype": "float16", "format": "NCHW",
                         "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float16", "format": "NCHW", "ori_format": "NCHW"},
                        0.001)
    get_op_support_info(
        {"shape": (2, 3, 2, 2, 2, 16), "ori_shape": (2, 3, 2, 2, 2, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (2, 3, 2, 2, 2, 16), "ori_shape": (2, 3, 2, 2, 2, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"}, 0.001)


def gen_bn_training_reduce_grad_case(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                     shape_batch_variance, dtype, dtype_others, case_name_val):
    return {"params": [
        {"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NC1HWC0", "ori_format": "NC1HWC0"},
        {"shape": shape_x_norm, "ori_shape": shape_x_norm, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_diff_scale, "ori_shape": shape_diff_scale, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_diff_offset, "ori_shape": shape_diff_offset, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_scale, "ori_shape": shape_scale, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NC1HWC0", "ori_format": "NC1HWC0"}],
            "case_name": case_name_val,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}


def gen_bn_training_reduce_grad_case1(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                      shape_batch_variance, dtype, dtype_others, case_name_val):
    return {"params": [{"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"},
                       {"shape": shape_x_norm, "ori_shape": shape_x_norm, "dtype": dtype_others, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"},
                       {"shape": shape_diff_scale, "ori_shape": shape_diff_scale, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_diff_offset, "ori_shape": shape_diff_offset, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_scale, "ori_shape": shape_scale, "dtype": dtype_others, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case5 = gen_bn_training_reduce_grad_case1((2, 3, 2, 2, 2, 16), (2, 3, 2, 2, 2, 16), (1, 1, 2, 1, 1, 16),
                                          (1, 1, 2, 1, 1, 16), (1, 1, 2, 1, 1, 16),
                                          (1, 1, 2, 1, 1, 16), "float16", "float32", "bn_training_reduce_grad_5")
case2 = gen_bn_training_reduce_grad_case((2, 3, 4, 2, 15), (2, 3, 4, 2, 15), (1, 3, 1, 1, 16), (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16), "float16", "float32", "bn_training_reduce_grad_2")
case3 = gen_bn_training_reduce_grad_case((2, 3, 4, 2, 19, 16, 1), (2, 3, 4, 2, 16), (1, 3, 1, 1, 16), (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16), "float16", "float32", "bn_training_reduce_grad_3")
case4 = gen_bn_training_reduce_grad_case((2, 3, 4, 2, 18), (2, 3, 4, 2, 16), (1, 3, 1, 1, 16), (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16), "float16", "float32", "bn_training_reduce_grad_4")
case1 = gen_bn_training_reduce_grad_case((2, 3, 4, 2, 18), (2, 3, 4, 2, 16), (1, 3, 1, 1, 16), (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16),
                                         (1, 3, 1, 1, 16), "float16", "float32", "bn_training_reduce_grad_1")


def test_op_select_format(test_arg):
    from impl.bn_training_reduce_grad import op_select_format
    op_select_format(
        {"shape": (2, 3, 2, 2, 2, 16), "ori_shape": (2, 3, 2, 2, 2, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (2, 3, 2, 2, 2, 16), "ori_shape": (2, 3, 2, 2, 2, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float16", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"}, 0.001)


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
