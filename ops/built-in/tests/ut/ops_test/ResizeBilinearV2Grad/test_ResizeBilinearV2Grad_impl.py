#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("ResizeBilinearV2Grad", None, None)

case1 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,257,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,257,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2, 3,25,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 3,25,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    {"shape": (2,3,1,1,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,3,1,1,16),"ori_format": "NHWC"},
                    True, False],
         "case_name": "resize_bilinear_v2_grad_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (3,3,2,256,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,256,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (3,3,2,256,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,256,16),"ori_format": "NHWC"},
                    {"shape": (3,3,33,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,33,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (2, 1, 768, 1152, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1, 768, 1152, 16),"ori_format": "NHWC"},
                    {"shape": (2, 1, 48, 72, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1, 48, 72, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (2, 1, 48, 72, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1, 48, 72, 16),"ori_format": "NHWC"},
                    {"shape": (2, 1, 48, 72, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1, 48, 72, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case10 = {"params": [{"shape": (2, 1, 48, 72, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 1, 48, 72, 16),"ori_format": "NHWC"},
                    {"shape": (32, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (32, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}     # line314 elif start
case11 = {"params": [{"shape": (33, 1, 48, 72, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 48, 72, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case12 = {"params": [{"shape": (33, 1, 48, 130, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 48, 130, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case13 = {"params": [{"shape": (33, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case14 = {"params": [{"shape": (33, 2, 2, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 2, 3, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case15 = {"params": [{"shape": (33, 2, 33, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 33, 3, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_15",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case16 = {"params": [{"shape": (33, 2, 33, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 33, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 3, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case17 = {"params": [{"shape": (33, 2, 32, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 32, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 3, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_17",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case18 = {"params": [{"shape": (33, 2, 2, 3, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 2, 3, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_18",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case19 = {"params": [{"shape": (33, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_19",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case20 = {"params": [{"shape": (33, 1, 2, 500, 70), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 2, 500, 70),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 500, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 500, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_20",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case21 = {"params": [{"shape": (33, 1, 2, 256, 128), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 2, 256, 128),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 256, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 256, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_21",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case22 = {"params": [{"shape": (33, 1, 48, 130, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 48, 130, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    True, True],
         "case_name": "resize_bilinear_v2_grad_22",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case23 = {"params": [{"shape": (3,3,2,260,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,260,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_23",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case24 = {"params": [{"shape": (3,3,2,260,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,260,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_24",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case25 = {"params": [{"shape": (33, 2, 33, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 33, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 650, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 650, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_25",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}    #line 1488 else
case26 = {"params": [{"shape": (33, 2, 33, 2, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 2, 33, 2, 16),"ori_format": "NHWC"},
                    {"shape": (3, 2, 2, 650, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 2, 2, 650, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, True],
         "case_name": "resize_bilinear_v2_grad_26",
         "expect": "success",
         "format_expect": [],
         "support_expect": True} 
case27 = {"params": [{"shape": (33, 1, 4, 240, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (33, 1, 4, 240, 16),"ori_format": "NHWC"},
                    {"shape": (3, 1, 1, 1, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (3, 1, 1, 1, 16),"ori_format": "NHWC"},
                    {"shape": (3,3,2,2,16), "dtype": "float32", "format": "NHWC", "ori_shape": (3,3,2,2,16),"ori_format": "NHWC"},
                    False, False],
         "case_name": "resize_bilinear_v2_grad_12",
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
ut_case.add_case(["Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend710", "Ascend910A"], case22)
ut_case.add_case(["Ascend710", "Ascend910A"], case23)
ut_case.add_case(["Ascend710", "Ascend910A"], case24)
ut_case.add_case(["Ascend710", "Ascend910A"], case25)
ut_case.add_case(["Ascend710", "Ascend910A"], case26)
ut_case.add_case(["Ascend710", "Ascend910A"], case27)
if __name__ == '__main__':
    ut_case.run(["Ascend710", "Ascend910A"])
    # ut_case.run("Ascend910")
    exit(0)








