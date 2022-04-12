# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import tuple_reduce_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("tuple_reduce_variable_shape", "varshape.test_tuple_reduce_variable_shape_impl")


def test_tuple_reduce_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)]
                }, 
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)]
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

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return ret_assert


test_funcs = [
    test_tuple_reduce_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
