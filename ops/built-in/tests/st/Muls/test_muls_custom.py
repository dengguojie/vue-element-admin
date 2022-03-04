#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from unittest.mock import patch
from unittest.mock import MagicMock

from impl.muls import muls_compute
from te import tvm
from te.lang.cce import auto_schedule
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce

def test_muls_compute_with_batchmatmul_1():
    x = tvm.placeholder((2, 2, 2, 16, 16), name="x", dtype="float16",
                        attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32)})
    tensor_x = tvm.compute((2, 2, 2, 16, 16), lambda *i: x(*i), name="tensor_x", tag="matmul",
                           attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32), "batch_shape": (2,)})
    output = {"shape": (2, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (2, 32, 32), "format": "FRACTAL_NZ"}
    muls_compute(tensor_x, output, 2.0, "muls_kernel")

def test_muls_compute_with_batchmatmul_2():
    x = tvm.placeholder((2, 2, 2, 16, 16), name="x", dtype="float16",
                        attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32)})
    tensor_x = tvm.compute((2, 2, 2, 16, 16), lambda *i: x(*i), name="tensor_x", tag="matmul",
                           attrs={'format': "FRACTAL_NZ", "ori_shape": (2, 32, 32), "batch_shape": (2,), "para_name": "muls"})
    output = {"shape": (2, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (2, 32, 32), "format": "FRACTAL_NZ"}
    muls_compute(tensor_x, output, 2.0, "muls_kernel")

if __name__ == '__main__':
    test_muls_compute_with_batchmatmul_1()
    test_muls_compute_with_batchmatmul_2()
