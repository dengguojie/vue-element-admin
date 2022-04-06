# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.reduce.reduce_pattern_parser import ReducePatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("reduce_pattern", "reduce.test_reduce_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_reduce_compute_graph_when_reduce_tensor_size_is_not_one(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.REDUCE: 2,
        ComputeType.ELEWISE: 1,
        ComputeType.ANY: 5,
    }

    matched = ReducePatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_match_with_reduce_compute_graph_when_reduce_tensor_size_is_one(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.REDUCE: 1,
        ComputeType.ELEWISE: 2,
        ComputeType.ANY: 5,
    }

    matched = ReducePatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_reduce_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 4,
        ComputeType.CAST: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 8,
    }

    matched = ReducePatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = ReducePatternParser(None, None, None).get_pattern()
    return pattern == Pattern.REDUCE


test_funcs = [
    test_match_with_reduce_compute_graph_when_reduce_tensor_size_is_not_one,
    test_match_with_reduce_compute_graph_when_reduce_tensor_size_is_one,
    test_match_with_not_reduce_compute_graph,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
