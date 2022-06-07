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
case13 = {"params": [{"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"},
                     {"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"},
                     {"shape": (47521, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 1),"ori_format": "NHWC"},
                     {"shape": (47521, 19978), "dtype": "float32", "format": "NHWC", "ori_shape": (47521, 19978),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_13",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case14 = {"params": [{"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"},
                     {"shape": (221, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (221, 1),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_14",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case15 = {"params": [{"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"},
                     {"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"},
                     {"shape": (24, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 1),"ori_format": "NHWC"},
                     {"shape": (24, 30000), "dtype": "float32", "format": "NHWC", "ori_shape": (24, 30000),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_15",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case16 = {"params": [{"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (14, 1), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 1),"ori_format": "NHWC"},
                     {"shape": (14, 11760), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_16",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case17 = {"params": [{"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (2125, 10881),"ori_format": "NHWC"},
                     {"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (14, 11760),"ori_format": "NHWC"},
                     {"shape": (2125,), "dtype": "float32", "format": "NHWC", "ori_shape": (2125,),"ori_format": "NHWC"},
                     {"shape": (2125, 10881), "dtype": "float32", "format": "NHWC", "ori_shape": (2125, 10881),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_17",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case18 = {"params": [{"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"},
                     {"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"},
                     {"shape": (17660,), "dtype": "float16", "format": "NHWC", "ori_shape": (17660,),"ori_format": "NHWC"},
                     {"shape": (17660, 8), "dtype": "float16", "format": "NHWC", "ori_shape": (17660, 8),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_18",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case19 = {"params": [{"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"},
                     {"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float16", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 768), "dtype": "float16", "format": "NHWC", "ori_shape": (32, 768),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_19",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case20 = {"params": [{"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float32", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 10), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 10),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_20",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case21 = {"params": [{"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"},
                     {"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float32", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"}],
          "addition_params": {"impl_mode": "high_precision"},
          "case_name": "softmax_cross_entropy_with_logits_21",
          "expect": "success",
          "format_expect": [],
          "support_expect": True}
case22 = {"params": [{"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"},
                     {"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"},
                     {"shape": (32,), "dtype": "float32", "format": "NHWC", "ori_shape": (32,),"ori_format": "NHWC"},
                     {"shape": (32, 17777), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 17777),"ori_format": "NHWC"}],
          "case_name": "softmax_cross_entropy_with_logits_22",
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

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case9)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case10)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case11)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case12)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case13)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case14)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case15)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case16)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case17)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case18)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case19)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case20)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case21)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case22)
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

ut_case.run("Ascend910")