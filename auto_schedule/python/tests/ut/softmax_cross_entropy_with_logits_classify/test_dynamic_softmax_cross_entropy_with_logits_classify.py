# # -*- coding:utf-8 -*-
import warnings
from sch_test_frame.ut import OpUT
import tbe
from tbe.dsl.base.classifier.with_reduce_softmax_cross_entropy_with_logits_classifier import WithReduceSoftmaxCrossEntropyWithLogitsClassifier
from tbe.dsl.base.operation import get_compile_info

warnings.filterwarnings("ignore")
ut_case = OpUT("softmax_cross_entropy_with_logits_classify", "softmax_cross_entropy_with_logits_classify.test_dynamic_softmax_cross_entropy_with_logits_classify")


def test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_may_tail_may(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 2147483647), (1, 2147483647)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 2147483647), (1, 2147483647)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(1, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'},
                       {'shape': [-1, -1], 'range': [(1, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(1, 2147483647), (1, 6544)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(1, 2147483647), (1, 6544)], 'mode': 'copy'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec1_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec1_and_cut'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec1'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec1'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec4_and_cut'},
                       {'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec4_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec4'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec4'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec2_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec2_and_cut'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec2'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec2'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec8_and_cut'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec8_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec8'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec8'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec9_and_cut'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec9_and_cut'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec9'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec9'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec6_and_cut'},
                       {'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec6_and_cut'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec6'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec6'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_not_tail_not(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 2147483647), (2, 2147483647)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 2147483647), (2, 2147483647)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'copy'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_not_tail_may(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 2147483647), (1, 2147483647)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 2147483647), (1, 2147483647)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (1, 6544)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (1, 6544)], 'mode': 'copy'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec2_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec2_and_cut'}],
                      [{'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec2'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec2'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec8_and_cut'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec8_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec8'},
                       {'shape': [-1, 1], 'range': [(2, 2147483647), (1, 1)], 'mode': 'vec8'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_may_tail_not(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 2147483647), (2, 2147483647)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 2147483647), (2, 2147483647)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(1, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'},
                       {'shape': [-1, -1], 'range': [(1, 2147483647), (6544, 2147483647)], 'mode': 'copy_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(1, 2147483647), (2, 6544)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(1, 2147483647), (2, 6544)], 'mode': 'copy'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec1_and_cut'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec1_and_cut'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec1'},
                       {'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec1'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (6544, 2147483647)], 'mode': 'vec4_and_cut'},
                       {'shape': [1, -1], 'range': [(1, 1), (6544, 2147483647)], 'mode': 'vec4_and_cut'}],
                      [{'shape': [-1, -1], 'range': [(2, 2147483647), (2, 6544)], 'mode': 'vec4'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 6544)], 'mode': 'vec4'}]]

        return ins == expect_ins



def test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_may_tail_may(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 5000), (1, 5000)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 5000), (1, 5000)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(1, 5000), (1, 5000)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(1, 5000), (1, 5000)], 'mode': 'copy'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec1'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec1'}],
                      [{'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec4'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec4'}],
                      [{'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec2'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec2'}],
                      [{'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec8'},
                       {'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec8'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec9'},
                       {'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec9'}],
                      [{'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec6'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec6'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_not_tail_not(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 5000), (2, 5000)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 5000), (2, 5000)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'copy'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_not_tail_may(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 5000), (1, 5000)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(2, 5000), (1, 5000)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(2, 5000), (1, 5000)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (1, 5000)], 'mode': 'copy'}],
                      [{'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec2'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec2'}],
                      [{'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec8'},
                       {'shape': [-1, 1], 'range': [(2, 5000), (1, 1)], 'mode': 'vec8'}]]

        return ins == expect_ins

def test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_may_tail_not(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 5000), (2, 5000)], "format": "ND"},
            {"dtype": "float32", "shape": [-1, -1], "range":[(1, 5000), (2, 5000)], "format": "ND"},
        ]
        ins = WithReduceSoftmaxCrossEntropyWithLogitsClassifier(inputs).classify()
        expect_ins = [[{'shape': [-1, -1], 'range': [(1, 5000), (2, 5000)], 'mode': 'copy'},
                       {'shape': [-1, -1], 'range': [(1, 5000), (2, 5000)], 'mode': 'copy'}],
                      [{'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec1'},
                       {'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec1'}],
                      [{'shape': [-1, -1], 'range': [(2, 5000), (2, 5000)], 'mode': 'vec4'},
                       {'shape': [1, -1], 'range': [(1, 1), (2, 5000)], 'mode': 'vec4'}]]

        return ins == expect_ins


ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_may_tail_may)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_not_tail_not)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_not_tail_may)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_not_cut_batch_may_tail_not)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_may_tail_may)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_not_tail_not)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_not_tail_may)
ut_case.add_cust_test_func(["Ascend910A", "Ascend310P3"], test_func=test_dynamic_softmax_cross_entropy_with_logits_classify_cut_batch_may_tail_not)

