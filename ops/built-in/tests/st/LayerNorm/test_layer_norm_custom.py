#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.layer_norm import get_op_support_info

def test_get_op_support_info_001():
    """
    test_get_op_support_info_001
    """
    print("enter test_get_op_support_info_001")
    input_x = {"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                  "ori_format": "ND", "param_type": "input"}
    input_gamma = {"shape": (62,), "dtype": "float32", "format": "ND", "ori_shape": (32, 62), "ori_format": "ND",
                    "param_type": "input"}
    input_beta = {"shape": (62,), "dtype": "float32", "format": "ND", "ori_shape": (32, 62), "ori_format": "ND",
                  "param_type": "input"}
    output_y = {"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                "ori_format": "ND",
                "param_type": "output"}
    output_mean = {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1), "ori_format": "ND",
                    "param_type": "output"}
    output_variance = {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1), "ori_format": "ND",
                        "param_type": "output"}
    begin_norm_axis = -1
    begin_params_axis = 1
    get_op_support_info(input_x, input_gamma, input_beta,
                        output_y, output_mean, output_variance,
                        begin_norm_axis, begin_params_axis)

if __name__ == '__main__':
    test_get_op_support_info_001()