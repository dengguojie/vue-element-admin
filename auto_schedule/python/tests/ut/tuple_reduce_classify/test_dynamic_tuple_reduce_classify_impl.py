# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe.dsl.base.classifier import classify_tuple_reduce
from tbe.common.context import op_context

warnings.filterwarnings("ignore")
ut_case = OpUT("tuple_reduce_classify", "tuple_reduce_classify.test_dynamic_tuple_reduce_classify_impl")

def test_static(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (24, 512, 102400)},
                  {"shape": (24, 512, 102400)},
                  [0, 1]]
        ins = classify_tuple_reduce(inputs, extra_params=None)
        expect_ins = [[{'shape': [12288, 102400], 'range':[(12288, 12288), (102400, 102400)]},
                       {'shape': [12288, 102400], 'range':[(12288, 12288), (102400, 102400)]},
                       [0]]]
        return ins == expect_ins

def test_ranked_dynamic(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1, -1, -1)},
                  {"shape": (-1, -1, -1)},
                  [0, 1]]
        ins = classify_tuple_reduce(inputs, extra_params=None)
        expect_ins = [[{'shape': [-1, -1], 'range':[(1, None), (1, None)]},
                       {'shape': [-1, -1], 'range':[(1, None), (1, None)]},
                       [0]]]
        return ins == expect_ins

ut_case.add_cust_test_func(["Ascend910A", "Ascend710"], test_static)
ut_case.add_cust_test_func(["Ascend910A", "Ascend710"], test_ranked_dynamic)
