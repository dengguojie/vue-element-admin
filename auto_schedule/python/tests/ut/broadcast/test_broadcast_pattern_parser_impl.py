# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.broadcast.broadcast_pattern_parser import BroadcastPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("broadcast_pattern", "broadcast.test_broadcast_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_broadcast_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 4,
        ComputeType.BROADCAST: 4,
        ComputeType.CAST: 1,
        ComputeType.ANY: 11,
    }

    matched = BroadcastPatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_broadcast_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 4,
        ComputeType.CAST: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.REDUCE: 1,
        ComputeType.ANY: 9,
    }

    matched = BroadcastPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = BroadcastPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.BROADCAST


test_funcs = [
    test_match_with_broadcast_compute_graph,
    test_match_with_not_broadcast_compute_graph,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
