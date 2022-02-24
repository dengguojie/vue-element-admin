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

def test_mul_compute_nz_nd_ubfusion_1():
    x = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
    y = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
    output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

def test_mul_compute_nz_nd_ubfusion_2():
    x = tvm.placeholder((2, 16, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (256, 32)})
    y = tvm.placeholder((256, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (256, 1)})
    output = {"shape": (2, 16, 16, 16), "dtype": "float16", "ori_shape": (256, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

def test_mul_compute_nd_nz_ubfusion_1():
    x = tvm.placeholder((1, 1, 1, 240), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (240,)})
    y = tvm.placeholder((15, 512, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (8192, 240)})
    output = {"shape": (15, 512, 16, 16), "dtype": "float16", "ori_shape": (8192, 240), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

def test_mul_compute_nd_nz_ubfusion_2():
    x = tvm.placeholder((256, 1), name="y", dtype="float16", attrs={'format': "ND", "ori_shape": (256, 1)})
    y = tvm.placeholder((2, 16, 16, 16), name="x", dtype="float16", attrs={'format': "FRACTAL_NZ", "ori_shape": (256, 32)})
    output = {"shape": (2, 16, 16, 16), "dtype": "float16", "ori_shape": (256, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
    mul_compute(x, y, output, False)

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

def test_op_select_format_3():
    x = {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"}
    y = {"shape": (256, 1), "dtype": "float16", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"}
    output = {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"}
    op_select_format(x, y, output)

def test_op_select_format_4():
    x = {"shape": (256, 1), "dtype": "float16", "format": "ND", "ori_shape": (256, 1), "ori_format": "ND"}
    y = {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"}
    output = {"shape": (256, 32), "dtype": "float16", "format": "ND", "ori_shape": (256, 32), "ori_format": "ND"}
    op_select_format(x, y, output)

if __name__ == '__main__':
    test_mul_compute_nz_nd_ubfusion_1()
    test_mul_compute_nz_nd_ubfusion_2()
    test_mul_compute_nd_nz_ubfusion_1()
    test_mul_compute_nd_nz_ubfusion_2()
    test_op_select_format_1()
    test_op_select_format_2()
    test_op_select_format_3()
    test_op_select_format_4()
