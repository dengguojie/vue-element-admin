# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe.dsl.base.classifier import classify_elewise
from tbe.common.context import op_context

warnings.filterwarnings("ignore")
ut_case = OpUT("max_inputs", "broadcast_classify.test_dynamic_broadcast_classify_impl")


def test_max_inputs(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1,),
                  "range": [(1, None)]}] * 71
        try:
            classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            print(e)
            return True
    return False


def test_no_intersection(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1,), "range": [(2, 10)]},
                  {"shape": (-1,), "range": [(11, 15)]}]
        try:
            classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            print(e)
            return True
    return False


def test_unknown_rank_error(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1, -2), "range": [(1, None), (1, None)]}]
        try:
            classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            print(e)
            return True
    return False


def test_unknown_rank(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (-2,), "dtype": "float16", "range": [(1, None)]}]
        try:
            classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            print(e)
            return False
    return True


test_func_list = [
    test_max_inputs,
    test_unknown_rank,
    test_unknown_rank_error,
    test_no_intersection
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)
