# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import concat_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("concat_variable_shape", "varshape.test_concat_variable_shape_impl")


def test_concat_variable_shape(_):
    with operation.dynamic():
        with operation.compute():
            intpus = [
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (2, 100)]
                },
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (4, 50)]
                },
            ]
            ret = vs.variable_shape([intpus])
            expected = [
                [
                    {
                        "name": "_dim_0_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_0_1",
                        "bound": (2, 100),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_dim_0_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_1",
                        "bound": (4, 50),
                        "category": var.Category.NORMAL,
                    },
                ],
            ]

            return var_util.equals_for_variable_shape(ret, expected)


test_funcs = [
    test_concat_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
