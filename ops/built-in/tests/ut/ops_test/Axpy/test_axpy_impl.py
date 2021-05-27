#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("Axpy", None, None)

ut_case.add_case("all", {
    "params": [{'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 16), 'ori_format': 'ND'},
               {'shape': (13, 15, 17, 16), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17 ,16), 'ori_format': 'ND'},
               2.0],
    "expect": RuntimeError
})

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

def calc_expect_func(input_x, input_y, output, alpha):
    shape_x=input_x.get("shape")
    shape_y=input_y.get("shape")
    output_arr=input_x.get("value")+input_y.get("value")*alpha
    return output_arr

# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [{'shape': (13, 15, 0, 0), 'dtype': 'float32',
#                 'format': 'FRACTAL_NZ', 'ori_shape': (13, 15, 0, 0),
#                 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (13, 15, 0, 0), 'dtype': 'float32',
#                 'format': 'FRACTAL_NZ', 'ori_shape': (13, 15, 0, 0),
#                 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (13, 15, 0, 0), 'dtype': 'float32',
#                 'format': 'FRACTAL_NZ', 'ori_shape': (13, 15, 0, 0),
#                 'ori_format': 'ND', "param_type": "output"},
#                2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

#ut_case.add_precision_case("all", {
#    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
#                'ori_format': 'ND', "param_type": "input"},
#               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
#                'ori_format': 'ND', "param_type": "input"},
#               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
#                'ori_format': 'ND', "param_type": "output"},
#               2.0],
#    "calc_expect_func": calc_expect_func,
#    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#})

ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (16,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16,), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

#ut_case.add_precision_case("all", {
#    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
#                'ori_format': 'ND', "param_type": "input"},
#               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
#                'ori_format': 'ND', "param_type": "input"},
#               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 32),
#                'ori_format': 'ND', "param_type": "output"},
#               2.0],
#    "calc_expect_func": calc_expect_func,
#    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#})

ut_case.add_precision_case("Ascend910", {
    "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (16, 1), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
                'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

#ut_case.add_precision_case("all", {
#    "params": [{'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#                'ori_format': 'ND', "param_type": "input"},
#               {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
#                'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "input"},
#               {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#                'ori_format': 'ND', "param_type": "output"},
#               2.0],
#    "calc_expect_func": calc_expect_func,
#    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#})

# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND', "param_type": "input"},
#         {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#          'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#          'ori_format': 'ND', "param_type": "input"},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND', "param_type": "output"},
#         2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })
# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [
#         {'shape': (16,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (16,),
#          'ori_format': 'ND', "param_type": "input"},
#         {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#          'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#          'ori_format': 'ND', "param_type": "input"},
#         {'shape': (16,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (16,),
#          'ori_format': 'ND', "param_type": "output"},
#         2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [{'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
#                 'ori_shape': (16, 1), 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#                 'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#                 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (16, 1), 'dtype': 'float32', 'format': 'ND',
#                 'ori_shape': (16, 1), 'ori_format': 'ND', "param_type": "output"},
#                2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })
# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [{'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
#                 'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (11, 12, 1, 1, 16, 16), 'dtype': 'float32',
#                 'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 16),
#                 'ori_format': 'ND', "param_type": "input"},
#                {'shape': (1, 1), 'dtype': 'float32', 'format': 'ND',
#                 'ori_shape': (1, 1), 'ori_format': 'ND', "param_type": "output"},
#                2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })
#
ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 16, 31),
                'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND', "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32',
                'format': 'FRACTAL_NZ', 'ori_shape': (11, 12, 13, 32),
                'ori_format': 'ND', "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_precision_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND', "param_type": "output"},
               1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND', "param_type": "input"},
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND', "param_type": "input"},
        {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND', "param_type": "output"},
        1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (13, 15, 17, 1), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 1), 'ori_format': 'ND', "param_type": "input"},
               {'shape': (13, 15, 17, 19), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND', "param_type": "output"},
               1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


#ut_case.add_precision_case("all", {
#    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
#                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "input"},
#               {'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
#                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "input"},
#               {'shape': (13, 15, 17, 19), 'dtype': 'float32', 'format': 'ND',
#                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "output"},
#               1.0],
#    "calc_expect_func": calc_expect_func,
#    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
#})


ut_case.add_precision_case("all", {
    "params": [{'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (13, 15, 17, 19, 21), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (13, 15, 17, 19, 21), 'ori_format': 'ND',  "param_type": "output"},
               1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'NHWC',  "param_type": "input"},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'NHWC',  "param_type": "input"},
               {'shape': (13, 15, 17, 19), 'dtype': 'int32', 'format': 'NHWC',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'NHWC',  "param_type": "output"},
               1.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
                'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
# TODO run error
# ut_case.add_precision_case("all", {
#     "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
#                 'ori_shape': (13, 15, 17, 19), 'ori_format': 'ND',  "param_type": "input"},
#                {'shape': (1, 1), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
#                 'ori_shape': (1, 1), 'ori_format': 'ND',  "param_type": "input"},
#                {'shape': (1, 1), 'dtype': 'float32', 'format': 'FRACTAL_NZ',
#                 'ori_shape': (1, 1), 'ori_format': 'ND',  "param_type": "output"},
#                2.0],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })
#
#
ut_case.add_precision_case("all", {
    "params": [{'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'NC1HWC0',
                'ori_shape': (11, 12, 2, 1, 16, 16), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'NC1HWC0',
                'ori_shape': (11, 12, 2, 1, 16, 16), 'ori_format': 'ND',  "param_type": "input"},
               {'shape': (11, 12, 2, 1, 16, 16), 'dtype': 'float32', 'format': 'NC1HWC0',
                'ori_shape': (11, 12, 2, 1, 16, 16), 'ori_format': 'ND',  "param_type": "output"},
               2.0],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910", simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
