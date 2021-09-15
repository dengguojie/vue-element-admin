#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("Axpy", "impl.axpy", "op_select_format")

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (16,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16,), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [{'shape': (1, 1), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("Ascend910", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (3, 3), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (3, 3), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 1), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
         'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {'shape': (16,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (16,),
         'ori_format': 'ND'},
        {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
         'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
         'ori_format': 'ND'},
        {'shape': (16,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (16,),
         'ori_format': 'ND'},
        2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 1), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 1), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (1, 3, 5, 7), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 3, 5, 7), 'ori_format': 'ND'},
               {'shape': (1, 3, 5, 7), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 3, 5, 7), 'ori_format': 'ND'},
               {'shape': (1, 3, 5, 7), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1, 3, 5, 7), 'ori_format': 'ND'},
               1.0],
    "expect": "success"
})


ut_case.add_case("Ascend910", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND'},
               {'shape': (3, 3), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (3, 3), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND'},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 1, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 1, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 1, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 1, 17, 19), 'ori_format': 'ND'},
               2.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND'},
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND'},
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND'},
        1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 1), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 1), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND'},
               1.0],
    "expect": "success"

 })

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (13, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (13, 16, 17, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (13, 16, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (13, 16, 17, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (13, 16, 17, 16), 'ori_format': 'NCHW'},
               {'shape': (13, 16, 17, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (13, 16, 17, 16), 'ori_format': 'NCHW'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'HWCN',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'HWCN'},
               {'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'HWCN',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'HWCN'},
               {'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'HWCN'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 16, 1, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 1, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 1, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 16, 1, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 1, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 1, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 16, 16, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 16, 16, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (32, 16, 1, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (32, 16, 1, 16), 'ori_format': 'NCHW'},
               {'shape': (32, 16, 16, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (32, 16, 16, 16), 'ori_format': 'NCHW'},
               {'shape': (32, 16, 16, 16), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (32, 16, 16, 15), 'ori_format': 'NCHW'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (16, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (16, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 15, 17, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 15, 17, 16), 'ori_format': 'NHWC'},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (16, 15, 17, 16), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16, 15, 17, 16), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16,), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16,), 'ori_format': 'NHWC'},
               {'shape': (16,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (16,), 'ori_format': 'ND'},
               {'shape': (16,), 'dtype': 'float16', 'format': 'NHWC',
                'ori_shape': (16,), 'ori_format': 'NHWC'},
               1.0],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 16, 15, 32), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 16, 15, 32), 'ori_format': 'NCHW'},
               {'shape': (16, 16, 1, 32), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 16, 1, 32), 'ori_format': 'NCHW'},
               {'shape': (16, 16, 15, 32), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 16, 15, 32), 'ori_format': 'NCHW'},
               1.0],
    "expect": "success"
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
