# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.varshape import transdata_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("transdata_variable_shape", "varshape.test_transdata_variable_shape_impl")


def test_transdata_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1, -1],
                    'format': 'NCHW',
                    'dtype': 'float16',
                    'range': [[1, None], [1, None], [1, None]],
                    'is_forward': True,
                    'transdata_category': 'general.forward'
                },
                [-1, -1, -1, 16],
                {
                    0: 0,
                    1: [1, 3],
                    2: 2
                }
            ]
            operation.add_compile_info_inner("_src_pad_mode", [0, 2, 0])
            operation.add_compile_info_inner("_src_pad_var", [1, 16, 1])

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_dim_0",
                        "bound": [1, None],
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1",
                        "bound": [1, None],
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_2",
                        "bound": [1, None],
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_dim_0",
                        "bound": [1, None],
                        "category": var.Category.NORMAL,
                    },
                    "floordiv(((_dim_1 + 16) - 1), 16)",
                    {
                        "name": "_dim_2",
                        "bound": [1, None],
                        "category": var.Category.NORMAL,
                    },
                    16
                ],
            ]

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return ret_assert


test_funcs = [
    test_transdata_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
