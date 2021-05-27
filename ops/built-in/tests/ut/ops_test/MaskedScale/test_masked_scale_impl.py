#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
ut_case = OpUT("MaskedScale", None, None)


#add cases
def MaskedScaleCases(Xshape, Xdtype, input_format, Mshape, Mdtype,
                     Yshape, Ydtype, attr_value=1.0, case_name="MaskedScale",
                     expect="success"):
    Case = {"params": [{"shape": Xshape, "dtype": Xdtype, "format": input_format, "ori_shape": Xshape,"ori_format": input_format},
                       {"shape": Mshape, "dtype": Mdtype, "format": input_format, "ori_shape": Mshape,"ori_format": input_format},
                       {"shape": Yshape, "dtype": Ydtype, "format": input_format, "ori_shape": Yshape,"ori_format": input_format},
                       attr_value],
            "case_name": case_name,
            "expect": expect,
            "format_expect": [input_format],
            "support_expect": True}
    return Case

# pylint: disable=unused-argument
def calc_expect_func(x, mask, y, value=1.0):
    input_x_value = x["value"]
    input_x_dtype = x["dtype"]
    input_mask_value = mask["value"]
    
    res = input_x_value * input_mask_value * value
    res = res.astype(input_x_dtype)
    return res

# do fill MaskedScalePrecisions param
def MaskedScalePrecisions(Xshape, Xdtype, Xformat, Mshape, Mdtype,
                          Yshape, Ydtype, attr_value=1.0, case_name="MaskedScalePrecisions"):
    Case = {"params": [{"shape": Xshape, "dtype": Xdtype, "format": Xformat, "ori_shape": Xshape,"ori_format": Xformat, "param_type":"input"},
                       {"shape": Mshape, "dtype": Mdtype, "format": Xformat, "ori_shape": Mshape,"ori_format": Xformat, "param_type":"input"},
                       {"shape": Yshape, "dtype": Ydtype, "format": Xformat, "ori_shape": Yshape,"ori_format": Xformat, "param_type":"output"},
                       attr_value],
            "expect": "success",
            "case_name": case_name,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)}
    return Case

# do test UT_AddCase(ut_case):
format_list = ["ND", "FRACTAL_NZ", "NC1HWC0", "FRACTAL_Z", "C1HWNCoC0"]
x_dtypes = ["float16","float32"]
mask_dtypes = ["float16","float32","int8"]
shape_data = [125,125,4]
shape_data_ex = [125,125,3]
for x_dtype in x_dtypes:
    for mask_dtype in mask_dtypes:
        for format_item in format_list:
            case = MaskedScaleCases(shape_data, x_dtype, format_item, shape_data, mask_dtype,
                                    shape_data, x_dtype,
                                     0.5, "MaskedScale_0"+x_dtype+mask_dtype+format_item, "success")
            ut_case.add_case(["Ascend310", "Ascend910A"],case)
            case = MaskedScaleCases(shape_data, x_dtype, format_item, shape_data, mask_dtype,
                                    shape_data_ex, x_dtype,
                                     0.5, "MaskedScale_1"+x_dtype+mask_dtype+format_item, RuntimeError)
            ut_case.add_case(["Ascend310", "Ascend910A"],case)
            case = MaskedScaleCases(shape_data, "int8", format_item, shape_data, mask_dtype,
                                    shape_data, x_dtype,
                                     0.5, "MaskedScale_2"+x_dtype+mask_dtype+format_item, RuntimeError)
            ut_case.add_case(["Ascend310", "Ascend910A"],case)

# do test UT_AddPrecisionCase(ut_case)
#for x_dtype_1 in x_dtypes:
#    for mask_dtype_1 in mask_dtypes:
#        # todo: test not support the case
#        if x_dtype_1 == "float16" and mask_dtype_1 == "float32":
#            continue
#        for format_item_1 in format_list:
#            test_case_name = "MaskedScalePrecision_"+x_dtype_1+"_"+mask_dtype_1+"_"+format_item_1
#            case = MaskedScalePrecisions(shape_data, x_dtype_1, format_item_1, shape_data, mask_dtype_1,
#                                         shape_data, x_dtype_1, 0.5, test_case_name)
#            ut_case.add_precision_case(["Ascend310"],case)

# pylint: disable=unused-argument
def test_op_select_format(test_arg):
    from impl.masked_scale import op_select_format

    x1 = {"ori_shape":(20, 28, 16, 16)}
    mask1 = {"ori_shape":(20, 28, 16, 16)}
    y1 = {"ori_shape":(20, 28, 16, 16)}
    value1 = 1.0
    op_select_format(x1, mask1, y1, value1, "masked_scale_0")

ut_case.add_cust_test_func(test_func=test_op_select_format)
