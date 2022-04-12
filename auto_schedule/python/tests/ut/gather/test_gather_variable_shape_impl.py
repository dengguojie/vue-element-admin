# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import gather_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("gather_variable_shape", "varshape.test_gather_variable_shape_impl")


def test_gather_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    "shape": (-1, -1, -1, -1),
                    "range": [(1, None), (1, None), (1, None), (1, None)],
                },
                {
                    "shape": (-1, -1),
                    "range": [(1, None), (1, None)],
                },
                2,
                1,
            ]

            operation.get_context().add("_gather_mode", "gather")

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_params_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_2",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_3",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_params_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_indices_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                2,
                1,
            ]

            axis_assert = crt_compute.get("_axis") == 2
            rank_assert = crt_compute.get("_rank") == 1
            params_shape_assert = crt_compute.get("_params_shape") == (-1, -1, -1, -1)
            indices_shape_assert = crt_compute.get("_indices_shape") == (-1, -1)

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return axis_assert and rank_assert and params_shape_assert and indices_shape_assert and ret_assert


def test_gather_nd_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    "shape": (-1, -1, -1, -1, -1),
                    "range": [(1, None), (1, None), (1, None), (1, None), (1, None)],
                },
                {
                    "shape": (-1, -1, 2),
                    "range": [(1, None), (1, None), (2, 2)],
                },
                1,
            ]

            operation.get_context().add("_gather_mode", "gather_nd")

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    {
                        "name": "_params_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_2",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_3",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_params_dim_4",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_params_dim_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_indices_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    2,
                ],
                1,
            ]

            axis_assert = crt_compute.get("_axis") == 0
            rank_assert = crt_compute.get("_rank") == 2
            params_shape_assert = crt_compute.get("_params_shape") == (-1, -1, -1, -1, -1)
            indices_shape_assert = crt_compute.get("_indices_shape") == (-1, -1, 2)

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return axis_assert and rank_assert and params_shape_assert and indices_shape_assert and ret_assert


test_funcs = [
    test_gather_variable_shape,
    test_gather_nd_variable_shape,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
