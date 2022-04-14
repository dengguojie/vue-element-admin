# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import slice_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("slice_variable_shape", "varshape.test_slice_variable_shape_impl")


def test_slice_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'dtype': 'float16', 
                    'range': [(1, None), (1, None)]
                }, 
                {
                    'shape': [2], 
                    'dtype': 'int32', 
                    'range': [(2, 2)]
                }, 
                {
                    'shape': [2], 
                    'dtype': 'int32', 
                    'range': [(2, 2)]
                }
            ]
            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_x_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_x_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_begin_dim_0",
                        "bound": (0, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_begin_dim_1",
                        "bound": (0, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    "(_begin_dim_0 + _size_dim_0)",
                    "(_begin_dim_1 + _size_dim_1)"
                ]
            ]

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return ret_assert


test_funcs = [
    test_slice_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
