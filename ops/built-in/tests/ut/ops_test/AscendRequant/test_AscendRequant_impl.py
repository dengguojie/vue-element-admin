#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from te import platform as cce_conf
from te import tvm

ut_case = OpUT("AscendRequant", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,1,1,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    True],
         "case_name": "ascend_requant_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (32,),"ori_format": "ND"},
                    {"shape": (1,1,4,4,32), "dtype": "int8", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_4",
         "expect": AttributeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2,1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "ND"},
                    {"shape": (1,1,1,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2,1,1,1,16,16), "dtype": "int32", "format": "NDC1HWC0", "ori_shape": (2,4,4),"ori_format": "NDHWC"},
                    {"shape": (1,1,1,1,1,16), "dtype": "uint64", "format": "NDC1HWC0", "ori_shape": (1,),"ori_format": "NDHWC"},
                    {"shape": (2,1,1,1,16,32), "dtype": "int8", "format": "NDC1HWC0", "ori_shape": (2,4,4),"ori_format": "NDHWC"},
                    False],
         "case_name": "ascend_requant_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (2,1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "ND"},
                    {"shape": (1,1,2,1,16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,1,1,16,32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "NC1HWC0"},
                    False],
         "case_name": "ascend_requant_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case8 = {"params": [{"shape": (2,1,1,1,16,16), "dtype": "int32", "format": "NDC1HWC0", "ori_shape": (2,4,4),"ori_format": "NDHWC"},
                    {"shape": (1,1,2,1,1,16), "dtype": "uint64", "format": "NDC1HWC0", "ori_shape": (1,),"ori_format": "NDHWC"},
                    {"shape": (2,1,1,1,16,32), "dtype": "int8", "format": "NDC1HWC0", "ori_shape": (2,4,4),"ori_format": "NDHWC"},
                    False],
         "case_name": "ascend_requant_8",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case9 = {"params": [{"shape": (1, 1, 1, 4, 4, 16), "dtype": "int32", "format": "NDC1HWC0", "ori_shape": (1, 1, 1, 4, 4, 16),"ori_format": "NDC1HWC0"},
                    {"shape": (1, 1, 1, 4, 4, 16), "dtype": "uint64", "format": "NDC1HWC0", "ori_shape": (1, 1, 1, 4, 4, 16),"ori_format": "NDC1HWC0"},
                    {"shape": (1, 1, 1, 4, 4, 32), "dtype": "int8", "format": "NDC1HWC0", "ori_shape": (1, 1, 1, 4, 4, 32),"ori_format": "NDC1HWC0"},
                    False],
         "case_name": "ascend_requant_9",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

def test_conv2d_vector(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.ascend_requant import ascend_requant_compute
    from impl import ascend_quant_util as util

    cce_conf.te_set_version('Ascend310P3')
    shape_in = (16, 1024, 7, 7)
    shape_w = (1024, 1024, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 31) // 32
    Ci0 = 32

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    fm = tvm.placeholder(shape_in_5HD, name='mad1', dtype="int8", attrs={'ori_format': 'NCHW',
                                                                         "conv_shape": (16, 1024, 16, 16),
                                                                         "true_conv_shape": shape_in,
                                                                         "invalid_data_rm_flag": 1,
                                                                         "remove_padded_column_in_next_op": 1
                                                                         })

    filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int8",
                               attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
    bias_tensor = None

    vdeq_v200 = tvm.placeholder(shape_scale, name="deq_tensor",
                                     attrs={'ori_shape': [Co1*Co0]}, dtype="uint64")

    conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0)
    ascend_requant_compute(conv_res, vdeq_v200, None, True)
    cce_conf.cce_conf.te_set_version("Ascend310")

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case7)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case8)
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"], case9)
ut_case.add_cust_test_func(test_func=test_conv2d_vector)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


