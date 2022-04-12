# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import norm_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("norm_variable_shape", "varshape.test_norm_variable_shape_impl")


def test_norm_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'common', 
                    'input_type': 0, 
                    'broadcast_axis': None, 
                    'norm_pattern': 4000
                },
            ]

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],                
            ]

            broadcast_axis_assert = crt_compute.get("_broadcast_axis") is None
            norm_pattern_assert = crt_compute.get("_norm_pattern") == 4000

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return broadcast_axis_assert and norm_pattern_assert and ret_assert


def test_norm_variable_shape_with_pattern_5006(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [1, -1], 
                    'range': [(1, 1), (1, None)], 
                    'mode': 'common', 
                    'input_type': 0, 
                    'broadcast_axis': None, 
                    'norm_pattern': 5006
                }, 
                {
                    'shape': [1, 1], 
                    'range': [(1, 1), (1, 1)], 
                    'mode': 'broadcast_axis_known', 
                    'input_type': 1, 
                    'broadcast_axis': [0, 1], 
                    'norm_pattern': 5006
                }, 
                {
                    'shape': [1, 1], 
                    'range': [(1, 1), (1, 1)], 
                    'mode': 'broadcast_axis_known', 
                    'input_type': 1, 
                    'broadcast_axis': [0, 1], 
                    'norm_pattern': 5006
                },
            ]

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    1,
                    {
                        "name": "_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [1, 1],
                [1, 1],
            ]

            broadcast_axis_assert = crt_compute.get("_broadcast_axis") == [0, 1]
            norm_pattern_assert = crt_compute.get("_norm_pattern") == 5006

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return broadcast_axis_assert and norm_pattern_assert and ret_assert


test_funcs = [
    test_norm_variable_shape,
    test_norm_variable_shape_with_pattern_5006,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
