#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AscendQuant", None, None)

case1 = {"params": [{"shape": (1, 2, 4, 4, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (1, 2, 4, 4, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 2, 4, 4, 16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 2, 4, 4, 16),"ori_format": "NC1HWC0"},
                    1.0, 0.0, False, "Floor"],
         "case_name": "ascend_quant_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1, 2, 4, 4), "dtype": "float32", "format": "NHWC", "ori_shape": (1, 2, 4, 4),"ori_format": "NHWC"},
                    {"shape": (1, 2, 4, 4), "dtype": "int8", "format": "NHWC", "ori_shape": (1, 2, 4, 4),"ori_format": "NHWC"},
                    1.0, 0.0, False, "Floor"],
         "case_name": "ascend_quant_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 2, 4, 4, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 2, 4, 4, 16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 2, 4, 4, 16),"ori_format": "NC1HWC0"},
                    1.0, 0.0, False, "Trunc"],
         "case_name": "ascend_quant_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1, 3, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1, 3, 4, 4, 16),"ori_format": "NC1HWC0"},
                    {"shape": (1, 3, 4, 4, 16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1, 3, 4, 4, 16),"ori_format": "NC1HWC0"},
                    1.0, 3.0, False, "Ceil"],
         "case_name": "ascend_quant_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (10,21,40,40,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,21,40,40,16),"ori_format": "NC1HWC0"},
                    {"shape": (10,21,40,40,16), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (10,21,40,40,16),"ori_format": "NC1HWC0"},
                    2.0, 0.0, False, "Round"],
         "case_name": "ascend_quant_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
