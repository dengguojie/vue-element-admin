#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from impl.exp import exp
from impl.util.platform_adapter import OpImplMode

def test_exp():
    input_x = {"shape": (3, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 3), "ori_format": "ND"}
    output_y = {"shape": (3, 3), "dtype": "float16", "format": "ND", "ori_shape": (3, 3), "ori_format": "ND"}
    base = -1.0
    scale = 1.0
    shift = 0.0
    kernel_name = "exp"
    impl_mode = OpImplMode.HIGH_PERFORMANCE
    exp(input_x, output_y, base, scale, shift, kernel_name, impl_mode)

if __name__ == '__main__':
    test_exp()
