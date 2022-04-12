# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import reduce_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("reduce_variable_shape", "varshape.test_reduce_variable_shape_impl")


def test_reduce_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1, -1], 
                    'range': [(1, None), (1, None), (1, None)], 
                    'mode': 'special', 
                    'rel_pos_to_reduce': 'before'
                },
                {
                    'shape': [1],
                    'value': [1], 
                    'rel_pos_to_reduce': 'axis'
                }
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
                    {
                        "name": "_dim_2",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [1]
            ]

            mode_assert = crt_compute.get("_mode") == "special"

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return mode_assert and ret_assert


test_funcs = [
    test_reduce_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
