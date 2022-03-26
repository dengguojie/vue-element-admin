#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import imp
from importlib import reload
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

def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.exp
    importlib.reload(sys.modules.get("impl.dynamic.exp"))

if __name__ == '__main__':
    test_exp()
    reload_check_support()
