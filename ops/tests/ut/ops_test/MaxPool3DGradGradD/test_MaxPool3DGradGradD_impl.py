#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPool3dGradGradD", None, None)


#=====================================Compiler==================================

case_compiler_window_eq_stride = {"params": [
                    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,16),"ori_format": "NDHWC"}, #orig_input
                    {"shape": (1,1,1,1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NDHWC"}, #orig_output
                    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,16),"ori_format": "NDHWC"}, #grad_grad
                    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,1,2,2,16),"ori_format": "NDC1HWC0"},#assist
                    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,2,2,16),"ori_format": "NDHWC"},#output
                    (1, 2, 2, 2, 1),#ksize
                    (1, 2, 2, 2, 1),#strides
                    (0,0,0,0,0,0), #pads
                    "NDHWC"
                    ],
         "case_name": "case_compiler_window_eq_stride",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


case_compiler_window_gt_stride = {"params": [
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #orig_input
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #orig_output
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #grad_grad
    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,1,2,2,16),"ori_format": "NDC1HWC0"},#assist
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #output
    (1, 2, 2, 2, 1),#ksize
    (1, 1, 1, 1, 1),#strides
    (0,0,0,0,0,1),  #pads
    "NDHWC"
],
    "case_name": "case_compiler_window_gt_stride",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}


case_compiler_window_lt_stride = {"params": [
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #orig_input
    {"shape": (1,1,1,1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NDHWC"},     #orig_output
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #grad_grad
    {"shape": (1,2,1,2,2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,2,1,2,2,16),"ori_format": "NDC1HWC0"},#assist
    {"shape": (1,3,1,3,3,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,3,3,3,16),"ori_format": "NDHWC"},     #output
    (1, 2, 2, 2, 1),#ksize
    (1, 3, 3, 3, 1),#strides
    (0,0,0,0,0,1), #pads
    "NDHWC"
],
    "case_name": "case_compiler_window_lt_stride",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}


case_compiler_huge_dhw = {"params": [
    {"shape": (1,20,1, 20,200,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,20,20,200,16),"ori_format": "NDHWC"},    #orig_input
    {"shape": (1,10,1, 10,100,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,10,10,100,16),"ori_format": "NDHWC"},    #orig_output
    {"shape": (1,20,1, 20,20,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape":  (1,20,20,20,16),"ori_format": "NDHWC"},     #grad_grad
    {"shape": (1,2, 1, 2, 2,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape":   (1,2,1,2,2,16),"ori_format": "NDC1HWC0"},   #assist
    {"shape": (1,20,1, 20,20,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape":  (1,20,20,20,16),"ori_format": "NDHWC"},     #output
    (1, 2, 2, 2, 1), #ksize
    (1, 2, 2, 2, 1), #strides
    (0,0,0,0,0,1),   #pads
    "NDHWC"
],
    "case_name": "case_compiler_huge_dhw",
    "expect": "success",
    "format_expect": [],
    "support_expect": True}

#=====================================precision==================================

def calc_expect_func(orig_input, orig_output, grad_grad, assist, output, ksize, strides, pads, data_format="NDHWC"):
    print(orig_input, orig_output, grad_grad, assist, output, ksize, strides, pads, data_format)
    return []

case_precision_window_eq_stride = {
    "params": [{"shape" : (1, 6, 1, 6, 6, 16), "format" : "NDC1HWC0",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 3, 3, 3, 3, 16), "format" : "NDC1HWC0",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 6, 1, 6, 6, 16), "format" : "NDC1HWC0",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 2, 1, 2, 2, 16), "format" : "NDC1HWC0",  "dtype" : "float16", "param_type": "input"},
               {"shape" : (1, 6, 1, 6, 6, 16), "format" : "NDC1HWC0",  "dtype" : "float16", "param_type": "output"},
               (2, 2, 2),         #kernels
               (2, 2, 2),         #strides
               (0,0,0,0,0,0),     #pads
               "NDHWC",           #data_format
               ],
    "case_name": "case_precision_window_eq_stride",
    "expect": "success",
    "calc_expect_func": calc_expect_func
}



ut_case.add_case("all", case_compiler_window_eq_stride)
ut_case.add_case("all", case_compiler_window_gt_stride)
ut_case.add_case("all", case_compiler_window_lt_stride)
ut_case.add_case("all", case_compiler_huge_dhw)
#ut_case.add_precision_case("all", case_precision_window_eq_stride)

