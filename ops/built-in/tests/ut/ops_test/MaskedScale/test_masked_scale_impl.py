#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("MaskedScale", None, None)

#add cases
def MaskedScaleCases(x_shape, x_dtype, format, mask_shape, mask_dtype,
                     y_shape, y_dtype, attr_value=1.0, case_name="MaskedScale",
                     expect="success"):
    case = {"params": [{"shape": x_shape, "dtype": x_dtype, "format": format, "ori_shape": x_shape,"ori_format": format},
                       {"shape": mask_shape, "dtype": mask_dtype, "format": format, "ori_shape": mask_shape,"ori_format": format},
                       {"shape": y_shape, "dtype": y_dtype, "format": format, "ori_shape": y_shape,"ori_format": format},
                       attr_value],
            "case_name": case_name,
            "expect": expect,
            "format_expect": ["ND"],
            "support_expect": True}
    return case

ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float16", "ND", (125, 125), "int8",
                                   (125, 125), "float16",
                                    0.5, "MaskedScale_0", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float32", "ND", (125, 125), "int8",
                                   (125, 125), "float32",
                                    0.5, "MaskedScale_1", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float16", "ND", (125, 125), "float16",
                                   (125, 125), "float16",
                                    0.5, "MaskedScale_2", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float32", "ND", (125, 125), "float16",
                                   (125, 125), "float32",
                                    0.5, "MaskedScale_3", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float16", "ND", (125, 125), "float32",
                                   (125, 125), "float16",
                                    0.5, "MaskedScale_4", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float32", "ND", (125, 125), "float32",
                                   (125, 125), "float32",
                                    0.5, "MaskedScale_5", "success"))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "float16", "ND", (125, 125), "int8",
                                   (1, 125), "float16",
                                    0.5, "MaskedScale_6", RuntimeError))
ut_case.add_case(["Ascend310", "Ascend910A"], 
                  MaskedScaleCases((125,125), "int8", "ND", (125, 125), "int8",
                                   (125, 125), "float32",
                                    0.5, "MaskedScale_7", RuntimeError))
#precision cases
def calc_expect_func(x, mask, y, value=1.0):
    input_x_value = x["value"]
    input_x_dtype = x["dtype"]
    input_mask_value = mask["value"]

    res = input_x_value * input_mask_value * value
    res = res.astype(input_x_dtype)
    return res

def MaskedScalePrecisions(x_shape, x_dtype, format, mask_shape, mask_dtype,
                          y_shape, y_dtype, attr_value=1.0):
    case = {"params": [{"shape": x_shape, "dtype": x_dtype, "format": format, "ori_shape": x_shape,"ori_format": format, "param_type":"input"},
                       {"shape": mask_shape, "dtype": mask_dtype, "format": format, "ori_shape": mask_shape,"ori_format": format, "param_type":"input"},
                       {"shape": y_shape, "dtype": y_dtype, "format": format, "ori_shape": y_shape,"ori_format": format, "param_type":"output"},
                       attr_value],
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)}
    return case

ut_case.add_precision_case(["Ascend310", "Ascend910A"], 
                           MaskedScalePrecisions((125,125), "float32", "ND", (125, 125), "float32",
                                                 (125, 125), "float32", 0.5))
ut_case.add_precision_case(["Ascend310", "Ascend910A"], 
                           MaskedScalePrecisions((125,125), "float16", "ND", (125, 125), "float16",
                                                 (125, 125), "float16", 0.5))
ut_case.add_precision_case(["Ascend310", "Ascend910A"], 
                           MaskedScalePrecisions((125,125), "float16", "ND", (125, 125), "int8",
                                                 (125, 125), "float16", 0.5))
