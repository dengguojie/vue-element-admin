#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("SoftmaxCrossEntropyLoss", None, None)

case1 = {"params": [{"shape": (5, 3, 6), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6),"ori_format": "ND"},
                    {"shape": (5, 6), "dtype": "int32", "format": "ND", "ori_shape": (5, 6),"ori_format": "ND"},
                    {"shape": (3,), "dtype": "float32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (5, 3, 6), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6),"ori_format": "ND"},
                    {"shape": (5, 3, 6), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6),"ori_format": "ND"},
                    0, "none"],
         "case_name": "softmax_cross_entropy_loss_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (5, 3), "dtype": "float32", "format": "ND", "ori_shape": (5, 3),"ori_format": "ND"},
                    {"shape": (5,), "dtype": "int32", "format": "ND", "ori_shape": (5,),"ori_format": "ND"},
                    {"shape": (3,), "dtype": "float32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (5, 3), "dtype": "float32", "format": "ND", "ori_shape": (5, 3),"ori_format": "ND"},
                    {"shape": (5, 3), "dtype": "float32", "format": "ND", "ori_shape": (5, 3),"ori_format": "ND"},
                    0, "none"],
         "case_name": "softmax_cross_entropy_loss_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
		 
case3 = {"params": [{"shape": (5, 3, 6, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6, 4),"ori_format": "ND"},
                    {"shape": (5, 6, 4), "dtype": "int32", "format": "ND", "ori_shape": (5, 6, 4),"ori_format": "ND"},
                    {"shape": (3,), "dtype": "float32", "format": "ND", "ori_shape": (3,),"ori_format": "ND"},
                    {"shape": (5, 3, 6, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6, 4),"ori_format": "ND"},
                    {"shape": (5, 3, 6, 4), "dtype": "float32", "format": "ND", "ori_shape": (5, 3, 6, 4),"ori_format": "ND"},
                    0, "none"],
         "case_name": "softmax_cross_entropy_loss_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (50, 3, 60, 4), "dtype": "float32", "format": "ND", "ori_shape": (50, 3, 60, 4),"ori_format": "ND"},
                    {"shape": (50, 60, 4), "dtype": "int32", "format": "ND", "ori_shape": (50, 6, 4),"ori_format": "ND"},
                    None,
                    {"shape": (50, 3, 60, 4), "dtype": "float32", "format": "ND", "ori_shape": (50, 3, 60, 4),"ori_format": "ND"},
                    {"shape": (50, 3, 60, 4), "dtype": "float32", "format": "ND", "ori_shape": (50, 3, 60, 4),"ori_format": "ND"},
                    0, "none"],
         "case_name": "softmax_cross_entropy_loss_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

if __name__ == "__main__":
    ut_case.run("Ascend910A")