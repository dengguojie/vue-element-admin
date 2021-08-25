#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("BnTrainingUpdateGrad", None, None)


def test_get_op_support_info(test_arg):
    from impl.bn_training_update_grad import get_op_support_info
    get_op_support_info({"shape": (2, 32, 5, 5), "ori_shape": (2, 32, 5, 5), "dtype": "float32", "format": "NCHW",
                         "ori_format": "NCHW"},
                        {"shape": (2, 32, 5, 5), "ori_shape": (2, 32, 5, 5), "dtype": "float32", "format": "NCHW",
                         "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NNCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        {"shape": (32), "ori_shape": (32), "dtype": "float32", "format": "NCHW", "ori_format": "NCHW"},
                        0.001)
    get_op_support_info(
        {"shape": (2, 1, 2, 5, 5, 16), "ori_shape": (2, 1, 2, 5, 5, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (2, 1, 2, 5, 5, 16), "ori_shape": (2, 1, 2, 5, 5, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"}, 0.001)


def gen_bn_training_update_grad_case(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, dtype, dtype_others,
                                     case_name_val):
    return {"params": [
        {"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NC1HWC0", "ori_format": "NC1HWC0"},
        {"shape": shape_x, "ori_shape": shape_x, "dtype": dtype, "format": "NC1HWC0", "ori_format": "NC1HWC0"},
        {"shape": shape_batch_mean, "ori_shape": shape_batch_mean, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"},
        {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others, "format": "NC1HWC0",
         "ori_format": "NC1HWC0"}],
        "case_name": case_name_val,
        "expect": RuntimeError,
        "format_expect": [],
        "support_expect": True}


def gen_bn_training_update_grad_case1(shape_grads, shape_x, shape_batch_mean, shape_batch_variance, dtype, dtype_others,
                                      case_name_val):
    return {"params": [{"shape": shape_grads, "ori_shape": shape_grads, "dtype": dtype, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"},
                       {"shape": shape_x, "ori_shape": shape_x, "dtype": dtype, "format": "NDC1HWC0",
                        "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_mean, "ori_shape": shape_batch_mean, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"},
                       {"shape": shape_batch_variance, "ori_shape": shape_batch_variance, "dtype": dtype_others,
                        "format": "NDC1HWC0", "ori_format": "NDC1HWC0"}],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = gen_bn_training_update_grad_case((2, 2, 2, 2, 16), (2, 2, 2, 2, 16), (1, 3, 1, 1, 16), (1, 2, 1, 1, 16),
                                         "float16", "float32",
                                         "bn_training_update_grad_1")
case2 = gen_bn_training_update_grad_case((2, 2, 2, 2, 16), (2, 2, 2, 2, 16), (1, 2, 2, 1, 16), (1, 2, 1, 1, 16),
                                         "float16", "float32",
                                         "bn_training_update_grad_2")
case3 = gen_bn_training_update_grad_case((2, 3, 4, 2, 19, 16, 1), (2, 3, 4, 2, 16), (1, 3, 1, 1, 16), (1, 3, 1, 1, 16),
                                         "float16", "float32",
                                         "bn_training_update_grad_3")
case4 = gen_bn_training_update_grad_case((2, 3, 4, 2, 16), (2, 3, 4, 2, 16), (1, 3, 1, 1, 16), (1, 3, 1, 1, 17),
                                         "float16", "float32",
                                         "bn_training_update_grad_4")
case5 = gen_bn_training_update_grad_case1((2, 1, 2, 5, 5, 16), (2, 1, 2, 5, 5, 16), (1, 1, 2, 1, 1, 16),
                                          (1, 1, 2, 1, 1, 16), "float16", "float32",
                                          "bn_training_update_grad_5")


def test_op_select_format(test_arg):
    from impl.bn_training_update_grad import op_select_format
    op_select_format(
        {"shape": (2, 1, 2, 5, 5, 16), "ori_shape": (2, 1, 2, 5, 5, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (2, 1, 2, 5, 5, 16), "ori_shape": (2, 1, 2, 5, 5, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"},
        {"shape": (1, 1, 2, 1, 1, 16), "ori_shape": (1, 1, 2, 1, 1, 16), "dtype": "float32", "format": "NDC1HWC0",
         "ori_format": "NDC1HWC0"}, 0.001)
    op_select_format({"shape": (1, 2, 1, 32), "ori_shape": (1, 2, 1, 32), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"},
                     {"shape": (1, 2, 1, 32), "ori_shape": (1, 2, 1, 32), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"},
                     {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"},
                     {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"},
                     {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"},
                     {"shape": (1, 2, 1, 1), "ori_shape": (1, 2, 1, 1), "dtype": "float32", "format": "NCHW",
                      "ori_format": "NCHW"}, 0.001)


ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
