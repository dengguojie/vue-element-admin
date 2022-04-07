# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.gather.gather_pattern_parser import GatherPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("gather_pattern", "gather.test_gather_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_gather_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 3,
        ComputeType.GATHER: 1,
        ComputeType.ANY: 4,
    }

    matched = GatherPatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_gather_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 3,
        ComputeType.GATHER: 1,
        ComputeType.CAST: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 6,
    }

    matched = GatherPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_match_with_all_placeholder(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 3,
        ComputeType.ANY: 3,
    }

    matched = GatherPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = GatherPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.GATHER


test_funcs = [
    test_match_with_gather_compute_graph,
    test_match_with_not_gather_compute_graph,
    test_match_with_all_placeholder,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
