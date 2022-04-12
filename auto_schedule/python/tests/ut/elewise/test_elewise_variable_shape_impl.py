# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import elewise_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("elewise_variable_shape", "varshape.test_elewise_variable_shape_impl")


def test_elewise_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    "shape": (-1,),
                    "range": [(1, None)],
                    "mode": "special",
                },
                {
                    "shape": (-1,),
                    "range": [(1, None)],
                    "mode": "special",
                },
            ]
            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_dim_0_0",
                        "bound": (1, None),
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

            const_shape_assert = crt_compute.get("_const_shape") is None
            pattern_assert = crt_compute.get("_pattern") is None
            mode_assert = crt_compute.get("_mode") == "special"

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return const_shape_assert and pattern_assert and mode_assert and ret_assert


def test_broadcast_variable_shape_with_aba_pattern(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    "shape": (-1, -1, -1),
                    "range": [(1, 256), (1, None), (100, 512)],
                    "mode": "special",
                    "pattern": "ABA"
                },
                {
                    "shape": (-1, -1, -1),
                    "range": [(1, 256), (1, None), (100, 512)],
                    "mode": "special",
                    "pattern": "ABA"
                },
            ]

            operation.get_context().add("_support_broadcast", True)

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_dim_0_0",
                        "bound": (1, 256),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_0",
                        "bound": (1, 2147483647),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_2_0",
                        "bound": (100, 512),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_dim_0_1",
                        "bound": (1, 256),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_1_1",
                        "bound": (1, 2147483647),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_dim_2_0",
                        "bound": (100, 512),
                        "category": var.Category.NORMAL,
                    },
                ],
            ]

            const_shape_assert = crt_compute.get("_const_shape") is None
            pattern_assert = crt_compute.get("_pattern") == "ABA"
            mode_assert = crt_compute.get("_mode") == "special"

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return const_shape_assert and pattern_assert and mode_assert and ret_assert


test_funcs = [
    test_elewise_variable_shape,
    test_broadcast_variable_shape_with_aba_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
