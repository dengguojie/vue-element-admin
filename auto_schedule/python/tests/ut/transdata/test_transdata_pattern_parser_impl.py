# # -*- coding:utf-8 -*-
import warnings

from tbe.dsl.unify_schedule.constants import ComputeType
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.vector.transdata.transdata_pattern_parser import TransdataPatternParser

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("transdata_pattern", "transdata.test_transdata_pattern_parser_impl")


# noinspection PyTypeChecker
def test_match_with_transdata_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 1,
        ComputeType.TRANSDATA: 1,
        ComputeType.ANY: 2,
    }

    matched = TransdataPatternParser(None, compute_type_size_map, None).match()
    return matched is True


# noinspection PyTypeChecker
def test_match_with_not_transdata_compute_graph(_):
    compute_type_size_map = {
        ComputeType.PLACEHOLDER: 2,
        ComputeType.ELEWISE: 4,
        ComputeType.CAST: 1,
        ComputeType.BROADCAST: 1,
        ComputeType.ANY: 8,
    }

    matched = TransdataPatternParser(None, compute_type_size_map, None).match()
    return matched is False


# noinspection PyTypeChecker
def test_get_pattern(_):
    pattern = TransdataPatternParser(None, None, None).get_pattern()
    return pattern == Pattern.TRANSDATA


test_funcs = [
    test_match_with_transdata_compute_graph,
    test_match_with_not_transdata_compute_graph,
    test_get_pattern,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
