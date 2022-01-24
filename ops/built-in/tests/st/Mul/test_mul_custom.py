#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import patch
from unittest.mock import MagicMock

from impl.mul import mul_compute
from impl.mul import op_select_format
from te import tvm
from te.lang.cce import auto_schedule
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce

def test_mul_compute_nz_nd_ubfusion():
    te_set_version("Ascend910")
    with cce():
        x = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
        y = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
        output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = mul_compute(x, y, output, False) 
        tensor_list = [x, y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "mul_compute_nz_nd_ubfusion",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_mul_compute_nd_nz_ubfusion():
    te_set_version("Ascend910")
    with cce():
        x = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
        y = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
        output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = mul_compute(x, y, output, False) 
        tensor_list = [x, y, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "mul_compute_nd_nz_ubfusion",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_op_select_format_1():
    x = {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240), "ori_format": "ND"}
    y = {"shape": (240,), "dtype": "float16", "format": "ND", "ori_shape": (240,), "ori_format": "ND"}
    output = {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240), "ori_format": "ND"}
    op_select_format(x, y, output)

def test_op_select_format_2():
    x = {"shape": (240,), "dtype": "float16", "format": "ND", "ori_shape": (240,), "ori_format": "ND"}
    y = {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240), "ori_format": "ND"}
    output = {"shape": (8192, 240), "dtype": "float16", "format": "ND", "ori_shape": (8192, 240), "ori_format": "ND"}
    op_select_format(x, y, output)

if __name__ == '__main__':
    test_mul_compute_nz_nd_ubfusion()
    test_mul_compute_nd_nz_ubfusion()
    test_op_select_format_1()
    test_op_select_format_2()