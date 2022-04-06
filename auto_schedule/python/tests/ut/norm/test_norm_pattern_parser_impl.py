# # -*- coding:utf-8 -*-
import warnings

import tbe
from tbe import tvm
from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.norm.norm_pattern_parser import NormPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("norm_pattern", "norm.test_norm_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_norm_compute_graph(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    reduce_sum_1 = tbe.dsl.reduce_sum(add_1, (1,), keepdims=True)
    broadcast_1 = tbe.dsl.broadcast(reduce_sum_1, shape)
    mul_1 = tbe.dsl.vmul(add_1, broadcast_1)

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 1,
        ComputeType.ELEWISE: 2,
        ComputeType.REDUCE: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 5,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1],
        ComputeType.ELEWISE: [add_1, mul_1],
        ComputeType.REDUCE: [reduce_sum_1],
        ComputeType.BROADCAST: [broadcast_1],
        ComputeType.ANY: [ph_1, add_1, mul_1, reduce_sum_1, broadcast_1],
    }

    matched = NormPatternParser([mul_1], compute_type_size_map, compute_type_tensor_map).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_norm_compute_graph(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    reduce_sum_1 = tbe.dsl.reduce_sum(add_1, (1,), keepdims=True)
    reduce_sum_2 = tbe.dsl.reduce_sum(reduce_sum_1, (1,), keepdims=True)

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 1,
        ComputeType.ELEWISE: 1,
        ComputeType.REDUCE: 2,
        ComputeType.ANY: 4,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1],
        ComputeType.ELEWISE: [add_1],
        ComputeType.REDUCE: [reduce_sum_1, reduce_sum_2],
        ComputeType.ANY: [ph_1, add_1, reduce_sum_1, reduce_sum_2],
    }

    matched = NormPatternParser([reduce_sum_2], compute_type_size_map, compute_type_tensor_map).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = NormPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.NORM


test_funcs = [
    test_match_with_norm_compute_graph,
    test_match_with_not_norm_compute_graph,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
