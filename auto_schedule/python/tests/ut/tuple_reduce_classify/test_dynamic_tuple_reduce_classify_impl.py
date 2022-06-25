# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe.dsl.classifier import tuple_reduce_classifier
from tbe.common.context import op_context

warnings.filterwarnings("ignore")
ut_case = OpUT("tuple_reduce_classify", "tuple_reduce_classify.test_dynamic_tuple_reduce_classify_impl")

def test_static(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (24, 512, 102400)},
                  {"shape": (24, 512, 102400)},
                  [0, 1]]
        ins = tuple_reduce_classifier.classify(inputs, extra_params=None)
        expect_ins = [[{'shape': [12288, 102400], 'range':[(12288, 12288), (102400, 102400)]},
                       {'shape': [12288, 102400], 'range':[(12288, 12288), (102400, 102400)]},
                       [0]]]
        return ins == expect_ins

def test_ranked_dynamic(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1, -1, -1, -1, -1)}, [0, 2, 3]]
        ins = tuple_reduce_classifier.classify(inputs, extra_params=None)
        expect_ins = [[{'shape': [-1, -1, -1, -1], 'range':[(1, None), (1, None), (1, None), (1, None)]}, [0, 2]],
                      [{'shape': [-1, -1, -1], 'range':[(1, None), (1, None), (1, None)]}, [0, 2]]]
        return ins == expect_ins

ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_static)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_ranked_dynamic)
