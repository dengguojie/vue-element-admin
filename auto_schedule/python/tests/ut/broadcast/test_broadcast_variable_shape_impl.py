# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import elewise_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

MAX_UNKNOWN_SHAPE_NUM = 2 ** 31 - 1


warnings.filterwarnings("ignore")
ut_case = OpUT("concat_variable_shape", "varshape.test_broadcast_variable_shape_impl")


def test_broadcast_variable_shape_same_input(_):
    with operation.dynamic():
        with operation.compute():
            intpus = [
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (1, None)]
                },
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (1, None)]
                },
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (1, None)]
                },
            ]
            operation.get_context().add("_same_input_shape_group", [[0], [1, 2]])
            operation.get_context().add("_support_broadcast", True)
            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_dim_0_0",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_0",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_dim_0_1",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_1",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_dim_0_1",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_1",
                        "bound": (1, MAX_UNKNOWN_SHAPE_NUM),
                        "category": var.Category.NORMAL,
                    },
                ],
            ]

            return var_util.equals_for_variable_shape(ret, expected)


test_funcs = [
    test_broadcast_variable_shape_same_input,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
