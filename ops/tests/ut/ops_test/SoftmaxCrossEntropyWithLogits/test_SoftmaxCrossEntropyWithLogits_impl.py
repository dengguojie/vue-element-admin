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

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)





