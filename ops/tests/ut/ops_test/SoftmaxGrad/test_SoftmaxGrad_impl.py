#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SoftmaxGrad", None, None)

case1 = {"params": [{"shape": (128, 255, 36), "dtype": "float16", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float16", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float16", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"}],
         "case_name": "softmax_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"},
                    {"shape": (128, 255, 36), "dtype": "float32", "format": "ND", "ori_shape": (128, 255, 36),"ori_format": "ND"}],
         "case_name": "softmax_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (16, 128), "dtype": "float16", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND"},
                    {"shape": (16, 128), "dtype": "float16", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND"},
                    {"shape": (16, 128), "dtype": "float16", "format": "ND", "ori_shape": (16, 128),"ori_format": "ND"}],
         "case_name": "softmax_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (123, 123), "dtype": "float32", "format": "ND", "ori_shape": (123, 123),"ori_format": "ND"},
                    {"shape": (123, 123), "dtype": "float32", "format": "ND", "ori_shape": (123, 123),"ori_format": "ND"},
                    {"shape": (123, 123), "dtype": "float32", "format": "ND", "ori_shape": (123, 123),"ori_format": "ND"}],
         "case_name": "softmax_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (28, ), "dtype": "float32", "format": "ND", "ori_shape": (28, ),"ori_format": "ND"},
                    {"shape": (28, ), "dtype": "float32", "format": "ND", "ori_shape": (28, ),"ori_format": "ND"},
                    {"shape": (28, ), "dtype": "float32", "format": "ND", "ori_shape": (28, ),"ori_format": "ND"}],
         "case_name": "softmax_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (11, 11), "dtype": "float16", "format": "ND", "ori_shape": (11, 11),"ori_format": "ND"},
                    {"shape": (11, 11), "dtype": "float16", "format": "ND", "ori_shape": (11, 11),"ori_format": "ND"},
                    {"shape": (11, 11), "dtype": "float16", "format": "ND", "ori_shape": (11, 11),"ori_format": "ND"}],
         "case_name": "softmax_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)


if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)



