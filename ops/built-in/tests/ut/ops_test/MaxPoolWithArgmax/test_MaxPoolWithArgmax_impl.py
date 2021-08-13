#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolWithArgmax", "impl.max_pool_with_argmax", "max_pool_with_argmax")

case1 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "VALID"],
         "case_name": "max_pool_with_arxmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888,),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,32,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,32,48,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,32,48,72,16),"ori_format": "NHWC"},
                    {"shape": (13888*16,), "dtype": "uint16", "format": "NHWC", "ori_shape": (13888*16,),"ori_format": "NHWC"},
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,72,72,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,72,72,16),"ori_format": "NHWC"},
                    {"shape": (1,1,36,36,16), "dtype": "float16", "format": "NHWC", "ori_shape": (1,1,36,36,16),"ori_format": "NHWC"},
                    {"shape": (11808,), "dtype": "uint16", "format": "NHWC", "ori_shape": (11808,),"ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (32,4,112,112,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,112,112,16),"ori_format": "NHWC"},
                    {"shape": (32,4,56,56,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,56,56,16),"ori_format": "NHWC"},
                    {"shape": (32,4,9,197,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (32,4,9,197,16),"ori_format": "NHWC"},
                    [1, 3, 3, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2,2,31,20,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,17,52,32),"ori_format": "NHWC"},
                    {"shape": (2,2,7,3,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,7,3,32),"ori_format": "NHWC"},
                    {"shape": (2,2,7,3,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,7,3,16),"ori_format": "NHWC"},
                    [1, 26, 8, 1],
                    [1, 5, 9, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,2,17,52,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,17,52,32),"ori_format": "NHWC"},
                    {"shape": (2,2,9,26,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,9,26,32),"ori_format": "NHWC"},
                    {"shape": (2,2,9,26,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,9,26,16),"ori_format": "NHWC"},
                    [1, 8, 23, 1],
                    [1, 2, 2, 1],
                    "SAME"],
         "case_name": "max_pool_with_arxmax_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

#def test_max_pool_with_argmax_resnet50(test_arg):
#
#    from impl.max_pool_with_argmax import max_pool_with_argmax
#
#    from te import platform as cce_conf
#
#    cce_conf.cce_conf.te_set_version("Ascend910A", core_type="VectorCore")
#
#    max_pool_with_argmax({"shape": (32,4,112,112,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,112,112,16),"ori_format": "NHWC"},
#
#                         {"shape": (32,4,56,56,16), "dtype": "float16", "format": "NHWC", "ori_shape": (32,4,56,56,16),"ori_format": "NHWC"},
#
#                         {"shape": (32,4,9,197,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (32,4,9,197,16),"ori_format": "NHWC"},
#
#                         [1, 3, 3, 1],
#
#                         [1, 2, 2, 1],
#
#                         "SAME")
#
#    cce_conf.cce_conf.te_set_version(test_arg)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
#ut_case.add_cust_test_func(test_func=test_max_pool_with_argmax_resnet50)

from impl.max_pool_with_argmax import check_supported

# pylint: disable=unused-argument,unused-variable
def test_check_support(test_arg):
    # input_x, output_y, output_argmax, ksize, strides, padding
    res = check_supported(
                    {"shape": (2,2,96,144,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,96,144,16),"ori_format": "NHWC"},
                    {"shape": (2,2,6,9,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,6,9,16),"ori_format": "NHWC"},
                    {"shape": (2,2,6,9,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,6,9,16),"ori_format": "NHWC"},
                    [1, 16, 16, 1],
                    [1, 16, 16, 1],
                    "VALID",
                    "max_pool_with_argmax_check_support_case_001")
    assert not res[0]
    res = check_supported(
                    {"shape": (2,2,16,8,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,16,8,32),"ori_format": "NHWC"},
                    {"shape": (2,2,2,1,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,1,32),"ori_format": "NHWC"},
                    {"shape": (2,2,2,1,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,1,32),"ori_format": "NHWC"},
                    [1, 8, 8, 1],
                    [1, 8, 8, 1],
                    "VALID",
                    "max_pool_with_argmax_check_support_case_002")
    assert not res[0]
    res = check_supported(
                    {"shape": (2,2,16,16,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,16,16,16),"ori_format": "NHWC"},
                    {"shape": (2,2,2,2,16), "dtype": "float16", "format": "NHWC", "ori_shape": (2,2,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,2,2,2,16), "dtype": "uint16", "format": "NHWC", "ori_shape": (2,2,2,2,16),"ori_format": "NHWC"},
                    [1, 8, 8, 1],
                    [1, 8, 8, 1],
                    "SAME",
                    "max_pool_with_argmax_check_support_case_003")
    assert res[0]

ut_case.add_cust_test_func(test_func=test_check_support)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
