
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from unittest.mock import MagicMock
from unittest.mock import patch
ut_case = OpUT("MaxPoolGradWithArgmax", None, None)

case1 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "VALID"],
         "case_name": "max_pool_grad_with_arxmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    {"shape": (32,1,14,86,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,14,86,16),"ori_format": "NHWC"},
                    {"shape": (1576960,), "dtype": "uint16", "format": "NHWC", "ori_shape": (1576960,),"ori_format": "NHWC"},
                    {"shape": (32,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,1,70,514,16),"ori_format": "NHWC"},
                    [1, 5, 8, 1],
                    [1, 5, 6, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,70,514,16),"ori_format": "NHWC"},
                    {"shape": (1,1,14,86,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,14,86,16),"ori_format": "NHWC"},
                    {"shape": (49280,), "dtype": "uint16", "format": "NHWC", "ori_shape": (49280,),"ori_format": "NHWC"},
                    {"shape": (1,1,70,514,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,70,514,16),"ori_format": "NHWC"},
                    [1, 5, 8, 1],
                    [1, 5, 6, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2,32,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,32,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888*16,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888*16,),"ori_format": "NHWC"},
                    {"shape": (2,32,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,96,144,16),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (256,4,112,112,16), "dtype": "float16", "format": "NC1HWC", "ori_shape": (256,112,112,64),"ori_format": "NHWC"},
                    {"shape": (256,4,56,56,16), "dtype": "float16", "format": "NC1HWC", "ori_shape": (256,56,56,64),"ori_format": "NHWC"},
                    {"shape": (256,4,9,197,16), "dtype": "uint16", "format": "NC1HWC", "ori_shape": (256,4,9,197,16),"ori_format": "NHWC"},
                    {"shape": (256,4,112,112,16), "dtype": "float16", "format": "NC1HWC", "ori_shape": (256,112,112,64),"ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,1,3,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,3,5,16),"ori_format": "NHWC"},
                    {"shape": (2,1,2,3,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,2,3,16),"ori_format": "NHWC"},
                    {"shape": (576,), "dtype": "uint16", "format": "NHWC", "ori_shape": (576,),"ori_format": "NHWC"},
                    {"shape": (2,1,3,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,3,5,16),"ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1_cut_h_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (2,20,40,800,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,20,40,800,16),"ori_format": "NHWC"},
                    {"shape": (2,20,40,400,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,20,40,400,16),"ori_format": "NHWC"},
                    {"shape": (2562560,), "dtype": "uint16", "format": "NHWC", "ori_shape": (2562560,),"ori_format": "NHWC"},
                    {"shape": (2,20,40,800,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,20,40,800,16),"ori_format": "NHWC"},
                    [1, 2, 2, 1],
                    [1, 1, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1_cut_w_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (1,1,3,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,3,5,16),"ori_format": "NHWC"},
                    {"shape": (1,1,2,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,2,5,16),"ori_format": "NHWC"},
                    {"shape": (96,), "dtype": "uint16", "format": "NHWC", "ori_shape": (96,),"ori_format": "NHWC"},
                    {"shape": (1,1,3,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,3,5,16),"ori_format": "NHWC"},
                    [1, 1, 3, 1],
                    [1, 2, 1, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1_cut_one_h_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (2,1,40,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,5,16),"ori_format": "NHWC"},
                    {"shape": (2,1,40,3,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,3,16),"ori_format": "NHWC"},
                    {"shape": (1152,), "dtype": "uint16", "format": "NHWC", "ori_shape": (1152,),"ori_format": "NHWC"},
                    {"shape": (2,1,40,5,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,5,16),"ori_format": "NHWC"},
                    [1, 2, 2, 1],
                    [1, 1, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1h_cut_h_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (2,1,40,1600,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,1600,16),"ori_format": "NHWC"},
                    {"shape": (2,1,40,800,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,800,16),"ori_format": "NHWC"},
                    {"shape": (256128,), "dtype": "uint16", "format": "NHWC", "ori_shape": (256128,),"ori_format": "NHWC"},
                    {"shape": (2,1,40,1600,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,40,1600,16),"ori_format": "NHWC"},
                    [1, 2, 2, 1],
                    [1, 1, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1h_cut_w_0",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (2,1,400,384,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,400,384,16),"ori_format": "NHWC"},
                    {"shape": (2,1,100,192,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,100,192,16),"ori_format": "NHWC"},
                    {"shape": (384320,), "dtype": "uint16", "format": "NHWC", "ori_shape": (384320,),"ori_format": "NHWC"},
                    {"shape": (2,1,400,384,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,400,384,16),"ori_format": "NHWC"},
                    [1, 5, 2, 1],
                    [1, 4, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1h_cut_one_h_1980",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case13 = {"params": [{"shape": (2,1,400,384,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,400,384,16),"ori_format": "NHWC"},
                    {"shape": (2,1,50,192,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,50,192,16),"ori_format": "NHWC"},
                    {"shape": (384640,), "dtype": "uint16", "format": "NHWC", "ori_shape": (384640,),"ori_format": "NHWC"},
                    {"shape": (2,1,400,384,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,1,400,384,16),"ori_format": "NHWC"},
                    [1, 10, 2, 1],
                    [1, 8, 2, 1],
                    "SAME"],
         "case_name": "max_pool_grad_with_arxmax_cut_nc1h_cut_one_h_1981",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310P3", "Ascend910A"], case1)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case2)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case3)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case4)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case5)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case6)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case7)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case8)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case9)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case10)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case11)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case12)
ut_case.add_case(["Ascend310P3", "Ascend910A"], case13)
from impl.max_pool_grad_with_argmax import check_supported

# pylint: disable=unused-argument,unused-variable
def test_check_support(test_arg):
    # x, grad, argmax, y, ksize, strides, padding, kernel_name
    res = check_supported(
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,6,9,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,6,9,16),"ori_format": "NHWC"},
                    {"shape": (2,2,6,9,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,6,9,16),"ori_format": "NHWC"},
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    [1, 16, 16, 1],
                    [1, 16, 16, 1],
                    "VALID",
                    "max_pool_grad_with_argmax_check_support_case_001")
    assert not res[0]
    res = check_supported(
                    {"shape": (2,2,16,8,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,16,8,32),"ori_format": "NHWC"},
                    {"shape": (2,2,2,1,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,1,32),"ori_format": "NHWC"},
                    {"shape": (2,2,2,1,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,1,32),"ori_format": "NHWC"},
                    {"shape": (2,2,16,8,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,16,8,32),"ori_format": "NHWC"},
                    [1, 8, 8, 1],
                    [1, 8, 8, 1],
                    "VALID",
                    "max_pool_grad_with_argmax_check_support_case_002")
    assert not res[0]
    res = check_supported(
                    {"shape": (2,2,16,16,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,16,16,16),"ori_format": "NHWC"},
                    {"shape": (2,2,2,2,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,2,2,2,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,2,16,16,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,16,16,16),"ori_format": "NHWC"},
                    [1, 8, 8, 1],
                    [1, 8, 8, 1],
                    "SAME",
                    "max_pool_grad_with_argmax_check_support_case_003")
    assert res[0]

ut_case.add_cust_test_func(test_func=test_check_support)


vals = {("tik.load3dv1",): False}
def side_effects(*args):
    return vals[args]
with patch("te.platform.cce_conf.api_check_support", MagicMock(side_effect=side_effects)):
    ut_case.run("Ascend910A")

#ut_case.run("Ascend910A")