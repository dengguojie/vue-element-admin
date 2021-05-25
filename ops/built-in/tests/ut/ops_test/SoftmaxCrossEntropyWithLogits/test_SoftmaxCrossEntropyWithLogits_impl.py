#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxCrossEntropyWithLogits", None, None)

case1 = {"params": [{"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float16", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float16", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float16", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_4",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (5, 2, 3), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2, 3),"ori_format": "NCHW"},
                    {"shape": (2, ), "dtype": "float32", "format": "NCHW", "ori_shape": (2, ),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"},
                    {"shape": (5, 2), "dtype": "float32", "format": "NCHW", "ori_shape": (5, 2),"ori_format": "NCHW"}],
         "case_name": "softmax_cross_entropy_with_logits_5",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"},
                    {"shape": (2, 1001), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1001),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (221, 8), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 8),"ori_format": "NHWC"},
                    {"shape": (8, ), "dtype": "float32", "format": "NHWC", "ori_shape": (8, ),"ori_format": "NHWC"},
                    {"shape": (221, ), "dtype": "float32", "format": "NHWC", "ori_shape": (221, ),"ori_format": "NHWC"},
                    {"shape": (221, 8), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 8),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"},
                    {"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"},
                    {"shape": (4771178,), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178,),"ori_format": "NHWC"},
                    {"shape": (4771178, 139), "dtype": "float32", "format": "NHWC", "ori_shape": (4771178, 139),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"},
                    {"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"},
                    {"shape": (16022,), "dtype": "float32", "format": "NHWC", "ori_shape": (16022,),"ori_format": "NHWC"},
                    {"shape": (16022, 16213), "dtype": "float32", "format": "NHWC", "ori_shape": (16022, 16213),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 1, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 1, 2, 32),"ori_format": "NHWC"},
                     {"shape": (99, 2, 2, 32), "dtype": "float32", "format": "NHWC", "ori_shape": (99, 2, 2, 32),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 1, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 1, 1, 4),"ori_format": "NHWC"},
                     {"shape": (64, 32, 1, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (64, 32, 1, 4),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 1, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 1, 104, 51),"ori_format": "NHWC"},
                     {"shape": (72, 9, 104, 51), "dtype": "float32", "format": "NHWC", "ori_shape": (72, 9, 104, 51),"ori_format": "NHWC"}],
         "case_name": "softmax_cross_entropy_with_logits_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
def test_get_op_support_info(test_arg):
    from impl.softmax_cross_entropy_with_logits import get_op_support_info
    get_op_support_info({"shape": (16, 16, 256, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 256, 16), "ori_format": "NCHW"},
                        {"shape": (16, 16, 256, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 256, 16), "ori_format": "NCHW"},
                        {"shape": (16, 1, 256, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 1, 256, 16), "ori_format": "NCHW"},
                        {"shape": (16, 16, 256, 16), "dtype": "float32", "format": "NCHW", "ori_shape": (16, 16, 256, 16), "ori_format": "NCHW"})
    get_op_support_info({"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                        {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (16, 1), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"})
    get_op_support_info({"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                        {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                        {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (16, 1), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"})
    get_op_support_info({"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1, ), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                        {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (16, 1), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"})
    get_op_support_info({"shape": (1, 1), "dtype": "float32", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                        {"shape": (16, 1), "dtype": "float32", "format": "ND", "ori_shape": (16, 1), "ori_format": "ND"},
                        {"shape": (16, 16), "dtype": "float32", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"})

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend920A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case12)
ut_case.add_cust_test_func(test_func=test_get_op_support_info)

def calc_expect_func(x1, x2, y1, y2):
    src_type = x1['dtype']
    inputArr1 = x1['value']
    inputArr2 = x2['value']
    if src_type == "float16":
        inputArr1 = inputArr1.astype("float32")
        inputArr2 = inputArr2.astype("float32")
    
    data_max = np.max(inputArr1, axis=-1, keepdims=True).astype("float32")
    data_sub = np.subtract(inputArr1, data_max).astype("float32")
    data_exp = np.exp(data_sub).astype("float32")
    data_sum = np.sum(data_exp, axis=-1, keepdims=True).astype("float32")
    data_softmax =  (data_exp / data_sum).astype("float32")
    data_log_tmp = np.log(data_sum).astype("float32")
    data_log = np.subtract(data_sub, data_log_tmp).astype("float32")
    data_mul = -np.multiply(inputArr2, data_log).astype("float32")
    if src_type == "float32":
        outputArr1 = np.sum(data_mul, axis=-1, keepdims=False)
        outputArr2 = np.subtract(data_softmax, inputArr2)
    else:
        outputArr1 = np.sum(data_mul, axis=-1, keepdims=False).astype("float16")
        outputArr2 = np.subtract(data_softmax, inputArr2).astype("float16")

