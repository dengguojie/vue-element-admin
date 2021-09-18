#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeBilinearV2GradD", "impl.resize_bilinear_v2_grad", "check_supported")

case1 = {"params": [{"shape": (2,3,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,3,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,25,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,257,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,3,1,16), "dtype": "float32", "format": "NWCH", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,25,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3,25,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    False],
         "case_name": "resize_bilinear_v2_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3,3,-1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (-2,), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (-2), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (-2), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,3,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1),"ori_format": "NHWC"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2),"ori_format": "NHWC"},
                    {"shape": (2,3,25,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,25,2,16),"ori_format": "NHWC"},
                    True],
         "case_name": "resize_bilinear_v2_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (2,3,1,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,1,1),"ori_format": "NCHW"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,2,2),"ori_format": "NCHW"},
                    {"shape": (2,3,25,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,25,2,16),"ori_format": "NCHW"},
                    True],
         "case_name": "resize_bilinear_v2_grad_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (2,3,15000,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,15000,1),"ori_format": "NCHW"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,2,2),"ori_format": "NCHW"},
                    {"shape": (2,3,25,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,25,2,16),"ori_format": "NCHW"},
                    True],
         "case_name": "resize_bilinear_v2_grad_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case11 = {"params": [{"shape": (2,3,0,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,0,1),"ori_format": "NCHW"},
                    {"shape": (2,3,2,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,2,2),"ori_format": "NCHW"},
                    {"shape": (2,3,25,16), "dtype": "float32", "format": "NCHW", "ori_shape": (2,3,25,2,16),"ori_format": "NCHW"},
                    True],
         "case_name": "resize_bilinear_v2_grad_10",
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
ut_case.add_case(["Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend710", "Ascend910A"], case11)
if __name__ == '__main__':
    ut_case.run(["Ascend710", "Ascend910A"])
    exit(0)

