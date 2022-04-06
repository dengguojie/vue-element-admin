# # -*- coding:utf-8 -*-
import warnings

import tbe
import te.lang.cce
from tbe import tvm
from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.tuple_reduce.tuple_reduce_pattern_parser import TupleReducePatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("tuple_reduce_pattern", "tuple_reduce.test_tuple_reduce_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_tuple_reduce_compute_graph(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = te.lang.cce.tuple_sum([add_1, mul_1], (1,))

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 2,
        ComputeType.REDUCE: 2,
        ComputeType.ANY: 6,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1, ph_2],
        ComputeType.ELEWISE: [add_1, mul_1],
        ComputeType.REDUCE: [sum_1, sum_2],
        ComputeType.ANY: [ph_1, ph_2, add_1, mul_1, sum_1, sum_2],
    }

    matched = TupleReducePatternParser([sum_1, sum_2], compute_type_size_map, compute_type_tensor_map).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_tuple_reduce_compute_graph(_):
    shape = (4, 5, 16)
    fp16 = "float16"

    ph_1 = tvm.placeholder(shape, dtype=fp16, name="ph_1")
    ph_2 = tvm.placeholder(shape, dtype=fp16, name="ph_2")
    add_1 = tbe.dsl.vadds(ph_1, 5)
    mul_1 = tbe.dsl.vmuls(ph_2, 4)
    sum_1, sum_2 = te.lang.cce.tuple_sum([add_1, mul_1], (1,), keepdims=True)
    add_2 = tbe.dsl.vadd(sum_1, sum_2)
    broadcast_1 = tbe.dsl.broadcast(add_2, shape)

    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 3,
        ComputeType.BROADCAST: 1,
        ComputeType.REDUCE: 2,
        ComputeType.ANY: 8,
    }

    compute_type_tensor_map = {
        ComputeType.PLACEHOLDER: [ph_1, ph_2],
        ComputeType.ELEWISE: [add_1, mul_1, add_2],
        ComputeType.REDUCE: [sum_1, sum_2],
        ComputeType.BROADCAST: [broadcast_1],
        ComputeType.ANY: [ph_1, ph_2, add_1, mul_1, sum_1, sum_2, add_2, broadcast_1],
    }

    matched = TupleReducePatternParser([broadcast_1], compute_type_size_map, compute_type_tensor_map).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = TupleReducePatternParser(None, None, None).get_pattern()
    return pattern == Pattern.TUPLE_REDUCE


test_funcs = [
    test_match_with_tuple_reduce_compute_graph,
    test_match_with_not_tuple_reduce_compute_graph,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
