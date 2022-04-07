# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.split.split_pattern_parser import SplitPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("split_pattern", "split.test_split_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_split_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 1,
        ComputeType.SPLIT: 1,
        ComputeType.ANY: 2,
    }

    matched = SplitPatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_split_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.CAST: 1,
        ComputeType.ELEWISE: 5,
        ComputeType.ANY: 8,
    }

    matched = SplitPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_match_with_all_placeholder(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 1,
        ComputeType.ANY: 1,
    }

    matched = SplitPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = SplitPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.SPLIT


test_funcs = [
    test_match_with_split_compute_graph,
    test_match_with_not_split_compute_graph,
    test_match_with_all_placeholder,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
