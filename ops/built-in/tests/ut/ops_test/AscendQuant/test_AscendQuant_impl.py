#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

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
case6 = {"params": [{"shape": (2,1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "ND"},
                    {"shape": (2,1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "ND"},
                    2.0, 0.0, False, "Round"],
         "case_name": "ascend_quant_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (10,21,40,40,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (10,21,40,40,16),"ori_format": "NC1HWC0"},
                    {"shape": (10,21,40,40,16), "dtype": "int4", "format": "NC1HWC0", "ori_shape": (10,21,40,40,16),"ori_format": "NC1HWC0"},
                    2.0, 0.0, False, "Round", 29],
         "case_name": "ascend_quant_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case6)
ut_case.add_case(["Ascend710"], case7)

#precision cases
def nhwc_data4Dto5D(inputData, channel0=16):
    Ftemp = np.shape(inputData)
    F = [Ftemp[0], np.int(np.ceil(Ftemp[3] * 1.0 / channel0)), Ftemp[1],
         Ftemp[2], channel0]
    outputData = np.zeros(F)
    for N in range(F[0]):
        for C1 in range(F[1]):
            for k in range(channel0):
                if (C1 * channel0 + k < Ftemp[3]):
                    outputData[N, C1, :, :, k] = inputData[N, :, :,
                                                 C1 * channel0 + k]
    return outputData

def nhwc_data5Dto4D(data5D, nhwcDims, channel0=16):
    nc1hwc0Dims = [nhwcDims[0], np.int(np.ceil(nhwcDims[3] * 1.0 / channel0)),
                   nhwcDims[1],
                   nhwcDims[2], channel0]
    outputData = np.zeros(nhwcDims, data5D.dtype)
    for N in range(nc1hwc0Dims[0]):
        for C1 in range(nc1hwc0Dims[1]):
            for k in range(channel0):
                if (C1 * channel0 + k < nhwcDims[3]):
                    outputData[N, :, :, C1 * channel0 + k] = data5D[N, C1, :, :,
                                                             k]
    return outputData

def calc_expect_func(x, y, scale, offset, sqrt_mode, round_mode):
    shape = x['shape']
    input_x = x['value']
    dtype = x['dtype']
    input_4d_shape = (shape[0],
                      shape[2],
                      shape[3],
                      shape[1] * shape[4]
                      )
    input_x_4d = nhwc_data5Dto4D(input_x, input_4d_shape)
    input_x_5d = nhwc_data4Dto5D(input_x_4d, 32)
    if dtype == "float32":
        input_x_5d = input_x_5d.astype("float16")
    scale_x_5d = input_x_5d * scale
    if sqrt_mode:
        scale_x_5d = scale_x_5d * scale
    offset_x_5d = scale_x_5d + offset
    if round_mode == "Round":
        out_data = offset_x_5d.round()
        out_data = out_data.astype("int8")
    elif round_mode == "Floor":
        out_data = np.floor(offset_x_5d)
        out_data = out_data.astype("int8")
    elif round_mode == "Ceil":
        out_data = np.ceil(offset_x_5d)
        out_data = out_data.astype("int8")
    elif round_mode == "Trunc":
        out_data = np.trunc(offset_x_5d)
        out_data = out_data.astype("int8")
    return out_data

ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1,2,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (1,1,4,4,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,4,4,32),"ori_format": "NC1HWC0", "param_type": "output"},
                                              1.0, 0.0, False, "Floor"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("Ascend910A", {"params": [{"shape": (1,2,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (1,1,4,4,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,4,4,32),"ori_format": "NC1HWC0", "param_type": "output"},
                                              1.0, 0.0, False, "Trunc"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })
ut_case.add_precision_case("all", {"params": [{"shape": (2,1,1,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2,1,1,16,16),"ori_format": "NC1HWC0", "param_type": "input","value_range":[-10,10]},
                                              {"shape": (2,1,1,16,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (2,1,1,16,32),"ori_format": "NC1HWC0", "param_type": "output"},
                                              1.0, 0.0, False, "Round"],
                                   "calc_expect_func": calc_expect_func,
                                   "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
                                   })

