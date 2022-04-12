# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape.vector import split_variable_shape as vs
from tbe.dsl.base import operation
from tbe.dsl.base import var

from sch_test_frame.ut import OpUT
from sch_test_frame.utils import var_util

warnings.filterwarnings("ignore")
ut_case = OpUT("split_variable_shape", "varshape.test_split_variable_shape_impl")


def test_split_variable_shape(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'split_general', 
                    'split_factor': 64
                },
                [-1, -1]
            ]
            operation.get_context().add("_avg_split", False)

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    "(floordiv(((_dim_0 + 64) - 1), 64)*64)",
                    {
                        "name": "_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_split_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_split_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
            ]

            mode_assert = crt_compute.get("_mode") == "split_general"

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return mode_assert and ret_assert


def test_split_variable_shape_with_avg(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'split_general', 
                    'split_factor': 64
                },
                [-1, -1]
            ]
            operation.get_context().add("_avg_split", True)

            ret = vs.variable_shape(intpus)
            expected = [
                [
                    "(floordiv(((_dim_0 + 64) - 1), 64)*64)",
                    {
                        "name": "_dim_1",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
                [
                    {
                        "name": "_split_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                    {
                        "name": "_split_0",
                        "bound": (1, None),
                        "category": var.Category.NORMAL,
                    },
                ],
            ]

            mode_assert = crt_compute.get("_mode") == "split_general"

            ret_assert = var_util.equals_for_variable_shape(ret, expected)

            return mode_assert and ret_assert


def test_split_variable_shape_with_input_len_error(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'split_general', 
                    'split_factor': 64
                },
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'split_general', 
                    'split_factor': 64
                },
                [-1, -1]
            ]

            try:
                vs.variable_shape(intpus)
            except RuntimeError as e:
                return "split variable shape requires two input parameters:" in e.args[0].get("detailed_cause")

            return False


def test_split_variable_shape_with_input_split_num_error(_):
    with operation.dynamic():
        with operation.compute() as crt_compute:
            intpus = [
                {
                    'shape': [-1, -1], 
                    'range': [(1, None), (1, None)], 
                    'mode': 'split_general', 
                    'split_factor': 64
                },
                [-1] * 64
            ]

            try:
                vs.variable_shape(intpus)
            except RuntimeError as e:
                return "split numbers error, split numbers must be" in e.args[0].get("detailed_cause")

            return False


test_funcs = [
    test_split_variable_shape,
    test_split_variable_shape_with_avg,
    test_split_variable_shape_with_input_len_error,
    test_split_variable_shape_with_input_split_num_error,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
