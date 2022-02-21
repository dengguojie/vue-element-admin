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
case7 = {"params": [{"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "ND", "ori_shape": (1,64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "ND", "ori_shape": (1,64,7,7),"ori_format": "NCHW"}],
         "case_name": "softmax_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,64,7,7),"ori_format": "NCHW"},
                    [1]],
         "case_name": "softmax_grad_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (1,16,2,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,16,32,32),"ori_format": "NCHW"},
                    {"shape": (1,16,2,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,16,32,32),"ori_format": "NCHW"},
                    {"shape": (1,16,2,2,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,16,32,32),"ori_format": "NCHW"},
                    [-1]],
         "case_name": "softmax_grad_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    [1]],
         "case_name": "softmax_grad_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    [1]],
         "case_name": "softmax_grad_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7),"ori_format": "NCHW"},
                    [-1]],
         "case_name": "softmax_grad_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case13 = {"params": [{"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    {"shape": (1,4,7,7,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (64,7,7),"ori_format": "NCHW"},
                    [-1]],
         "case_name": "softmax_grad_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend710", "Ascend910A"], case13)


from impl.softmax_grad import op_select_format

def test_softmax_grad_op_select_format_000(test_arg):
    """
    test_softmax_grad_op_select_format_000
    """
    op_select_format(
        {
            "shape": (4096, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (4096, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (4096, 4096),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        -1,
    )

def test_softmax_grad_op_select_format_001(test_arg):
    """
    test_softmax_grad_op_select_format_001
    """
    op_select_format(
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        {
            "shape": (32, 32, 4096),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (4096, 4096),
            "ori_format": "NHWC"
        },
        -1,
    )

ut_case.add_cust_test_func(test_func=test_softmax_grad_op_select_format_000)
ut_case.add_cust_test_func(test_func=test_softmax_grad_op_select_format_001)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)



