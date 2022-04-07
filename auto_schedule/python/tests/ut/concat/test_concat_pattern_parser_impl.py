# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.concat.concat_pattern_parser import ConcatPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("concat_pattern", "concat.test_concat_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_concat_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 5,
        ComputeType.CONCAT: 1,
        ComputeType.ANY: 6,
    }

    matched = ConcatPatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_concat_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 5,
        ComputeType.CONCAT: 1,
        ComputeType.CAST: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 8,
    }

    matched = ConcatPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_match_with_all_placeholder(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 5,
        ComputeType.ANY: 5,
    }

    matched = ConcatPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = ConcatPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.CONCAT


test_funcs = [
    test_match_with_concat_compute_graph,
    test_match_with_not_concat_compute_graph,
    test_match_with_all_placeholder,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
