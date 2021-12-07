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
            ins = classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001', 'detailed_cause': 'more than 70 input are not supported'}
            return error_message == e.args[0]
    return False


def test_no_intersection(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1,), "range": [(2, 10)]},
                  {"shape": (-1,), "range": [(11, 15)]}]
        try:
            ins = classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001', 'detailed_cause': 'input shape error, shape range no intersection'}
            return error_message == e.args[0]
    return False


def test_unknown_rank_error(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1, -2), "range": [(1, None), (1, None)]}]
        try:
            ins = classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'if the shape contains -2, it must be [-2] or (-2,)'}
            return error_message == e.args[0]
    return False


def test_unknown_rank(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (-2,), "dtype": "float16", "range": [(1, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [
            [{'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']}],
            [{'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['common', 'broadcast']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['common', 'broadcast']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['common', 'broadcast']}],
            [{'shape': (-1, -1, -1), 'range': ((1, 2147483647), (1, 2147483647), (1, 2147483647)),
              'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']},
             {'shape': (-1, -1, -1), 'range': ((1, 2147483647), (1, 2147483647), (1, 2147483647)),
              'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']},
             {'shape': (-1, -1, -1), 'range': ((1, 2147483647), (1, 2147483647), (1, 2147483647)),
              'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']}],
            [{'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['broadcast', 'common']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['broadcast', 'common']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
              'mode': 'special', 'pattern': ['broadcast', 'common']}],
            [{'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast']}],
            [{'shape': [-1, -1, -1, -1, -1, -1, -1, -1],
              'range': [(1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647),
                        (1, 2147483647), (1, 2147483647), (1, 2147483647)], 'support_broadcast': True,
              'mode': 'original'},
             {'shape': [-1, -1, -1, -1, -1, -1, -1, -1],
              'range': [(1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647),
                        (1, 2147483647), (1, 2147483647), (1, 2147483647)], 'support_broadcast': True,
              'mode': 'original'},
             {'shape': [-1, -1, -1, -1, -1, -1, -1, -1],
              'range': [(1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647), (1, 2147483647),
                        (1, 2147483647), (1, 2147483647), (1, 2147483647)], 'support_broadcast': True,
              'mode': 'original'}],
            [{'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'},
             {'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'},
             {'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'}],
            [{'shape': [-1, -1, -1, -1, -1, -1, -1],
              'range': [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None), (1, None)],
              'support_broadcast': True, 'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
             {'shape': [-1, -1, -1, -1, -1, -1, -1],
              'range': [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None), (1, None)],
              'support_broadcast': True, 'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
             {'shape': [-1, -1, -1, -1, -1, -1, -1],
              'range': [(1, None), (1, None), (1, None), (1, None), (1, None), (1, None), (1, None)],
              'support_broadcast': True, 'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']}]]
        return ins == except_ins


def test_no_intersection_mul(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1,), "range": [(2, 10)]},
                  {"shape": (-1,), "range": [(11, 15)]},
                  {"shape": (-1,), "range": [(16, 20)]}]
        try:
            classify_elewise(inputs, support_broadcast=True)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001', 'detailed_cause': 'input shape error, shape range no intersection'}
            return error_message == e.args[0]
    return False


def test_const_broadcast_mul(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1, -1), "range": [(1, None), (2, 10)]},
                  {"shape": (-1, -1), "range": [(1, None), (2, 15)]},
                  {"shape": (-1, -1), "range": [(1, None), (2, 20)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [
            [{'shape': (-1,), 'range': ((2, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((2, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((2, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']}],
            [{'shape': (-1, -1), 'range': ((1, 2147483647), (2, 10)), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast', 'common']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (2, 10)), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast', 'common']},
             {'shape': (-1, -1), 'range': ((1, 2147483647), (2, 10)), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['broadcast', 'common']}]]
        return ins == except_ins


def test_dim_length_1_mul(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-1,), "range": [(1, None)]},
                  {"shape": (-1,), "range": [(1, None)]},
                  {"shape": (-1,), "range": [(1, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [
            [{'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']}], [
                {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
                 'pattern': ['broadcast']},
                {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
                 'pattern': ['broadcast']},
                {'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
                 'pattern': ['broadcast']}]]
        return ins == except_ins


def test_mix_unknown_rank(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (30, -1, -1,), "dtype": "float16", "range": [(30, 30), (1, None), (1, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [
            [{'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']},
             {'shape': (-1,), 'range': ((30, 2147483647),), 'support_broadcast': True, 'mode': 'special',
              'pattern': ['common']}], [
                {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                 'mode': 'special', 'pattern': ['common', 'broadcast']},
                {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                 'mode': 'special', 'pattern': ['common', 'broadcast']}], [
                {'shape': (30, -1, -1), 'range': ((30, 30), (1, 2147483647), (1, 2147483647)),
                 'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']},
                {'shape': (30, -1, -1), 'range': ((30, 30), (1, 2147483647), (1, 2147483647)),
                 'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']}], [
                {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                 'mode': 'special', 'pattern': ['broadcast', 'common']},
                {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                 'mode': 'special', 'pattern': ['broadcast', 'common']}], [
                {'shape': [1], 'range': [(1, 1)], 'support_broadcast': True, 'mode': 'special_scalar',
                 'pattern': ('scalar', 'broadcast')},
                {'shape': [-1], 'range': [(1, None)], 'support_broadcast': True, 'mode': 'special_scalar',
                 'pattern': ('scalar', 'broadcast')}], [
                {'shape': [-1, -1, -1, -1], 'range': [(1, 2147483647), (1, 30), (1, 2147483647), (1, 2147483647)],
                 'support_broadcast': True, 'mode': 'original'},
                {'shape': [1, 30, -1, -1], 'range': [(1, 1), (30, 30), (1, 2147483647), (1, 2147483647)],
                 'support_broadcast': True, 'mode': 'original'}],
            [{'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'},
             {'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'}],
            [{'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
              'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
             {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
              'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']}]]
        return ins == except_ins


def test_empty_shape(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (30, 0, -1), "dtype": "float16", "range": [(30, 30), (0, 0), (1, None)]},
                  {"shape": (30, -1, -1,), "dtype": "float16", "range": [(30, 30), (1, None), (0, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [[{'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'},
                       {'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'}]]
        return ins == except_ins


def test_mix_empty_shape(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (-2,), "dtype": "float16", "range": [(1, None)]},
                  {"shape": (30, -1, -1,), "dtype": "float16", "range": [(30, 30), (1, None), (0, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [[{'shape': (-1,), 'range': ((1, 2147483647),), 'support_broadcast': True, 'mode': 'special',
                        'pattern': ['common']},
                       {'shape': (-1,), 'range': ((30, 2147483647),), 'support_broadcast': True, 'mode': 'special',
                        'pattern': ['common']}], [
                          {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                           'mode': 'special', 'pattern': ['common', 'broadcast']},
                          {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                           'mode': 'special', 'pattern': ['common', 'broadcast']}], [
                          {'shape': (30, -1, -1), 'range': ((30, 30), (1, 2147483647), (1, 2147483647)),
                           'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']},
                          {'shape': (30, -1, -1), 'range': ((30, 30), (1, 2147483647), (1, 2147483647)),
                           'support_broadcast': True, 'mode': 'special', 'pattern': ['common', 'broadcast', 'common']}],
                      [{'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                        'mode': 'special', 'pattern': ['broadcast', 'common']},
                       {'shape': (-1, -1), 'range': ((1, 2147483647), (1, 2147483647)), 'support_broadcast': True,
                        'mode': 'special', 'pattern': ['broadcast', 'common']}], [
                          {'shape': [1], 'range': [(1, 1)], 'support_broadcast': True, 'mode': 'special_scalar',
                           'pattern': ('scalar', 'broadcast')},
                          {'shape': [-1], 'range': [(1, None)], 'support_broadcast': True, 'mode': 'special_scalar',
                           'pattern': ('scalar', 'broadcast')}], [{'shape': [-1, -1, -1, -1],
                                                                   'range': [(1, 2147483647), (1, 30), (1, 2147483647),
                                                                             (1, 2147483647)],
                                                                   'support_broadcast': True, 'mode': 'original'},
                                                                  {'shape': [1, 30, -1, -1],
                                                                   'range': [(1, 1), (30, 30), (1, 2147483647),
                                                                             (1, 2147483647)],
                                                                   'support_broadcast': True, 'mode': 'original'}],
                      [{'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'},
                       {'shape': (0,), 'range': [(0, 0)], 'support_broadcast': True, 'mode': 'empty'}], [
                          {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
                           'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
                          {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
                           'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']}]]
        return ins == except_ins


def test_update_pattern(_):
    with op_context.OpContext("dynamic"):
        inputs = [{"shape": (2, 5, -1, -1), "dtype": "float16", "range": [(2, 2), (5, 5), (1, 8), (1, 1)]},
                  {"shape": (2, -1, -1, -1), "dtype": "float16", "range": [(2, 2), (1, None), (1, None), (1, None)]},
                  {"shape": (-1, 1, -1, -1), "dtype": "float16", "range": [(1, None), (1, 1), (1, None), (1, None)]},
                  {"shape": (1, -1, -1, -1), "dtype": "float16", "range": [(1, 1), (1, None), (1, None), (1, None)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [[
            {'shape': (-1, -1),
             'range': ((1, 10), (1, 8)), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast', 'common']},
            {'shape': (-1, -1), 'range': ((1, 10), (1, 2147483647)), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast', 'common']},
            {'shape': (1, -1), 'range': ((1, 1), (1, 2147483647)), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast', 'common']},
            {'shape': (1, -1), 'range': ((1, 1), (1, 2147483647)), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast', 'common']}], [
            {'shape': (-1,), 'range': ((10, 80),), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast']},
            {'shape': (-1,), 'range': ((2, 2147483647),), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast']},
            {'shape': (1,), 'range': ((1, 1),), 'support_broadcast': True, 'mode': 'special', 'pattern': ['broadcast']},
            {'shape': (1,), 'range': ((1, 1),), 'support_broadcast': True, 'mode': 'special',
             'pattern': ['broadcast']}], [
            {'shape': [2, 5, -1, 1], 'range': [(2, 2), (5, 5), (1, 8), (1, 1)], 'support_broadcast': True,
             'mode': 'original'},
            {'shape': [2, -1, -1, -1], 'range': [(2, 2), (1, 5), (1, 2147483647), (1, 2147483647)],
             'support_broadcast': True, 'mode': 'original'},
            {'shape': [-1, 1, -1, -1], 'range': [(1, 2), (1, 1), (1, 2147483647), (1, 2147483647)],
             'support_broadcast': True, 'mode': 'original'},
            {'shape': [1, -1, -1, -1], 'range': [(1, 1), (1, 5), (1, 2147483647), (1, 2147483647)],
             'support_broadcast': True, 'mode': 'original'}], [
            {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
             'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
            {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
             'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
            {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
             'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']},
            {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'support_broadcast': True,
             'mode': 'special', 'pattern': ['UNKNOWN', 'UNKNOWN', 'UNKNOWN']}]]
        return ins == except_ins

def test_mul_input_static(_):
    with op_context.OpContext("static"):
        inputs = [{"shape": (7, 1), "dtype": "float16", "range": [(7, 7), (1, 1)]},
                  {"shape": (7, 2), "dtype": "float16", "range": [(7, 7), (2, 2)]},
                  {"shape": (7, 2), "dtype": "float16", "range": [(7, 7), (2, 2)]}]
        ins = classify_elewise(inputs, support_broadcast=True)
        except_ins = [[{'shape': [7, 1], 'range': [(7, 7), (1, 1)], 'const_shape':[7, 1], 'mode': 'const','support_broadcast': True},
                       {'shape': [7, 2], 'range': [(7, 7), (2, 2)], 'const_shape':[7, 2], 'mode': 'const','support_broadcast': True},
                       {'shape': [7, 2], 'range': [(7, 7), (2, 2)], 'const_shape':[7, 2], 'mode': 'const','support_broadcast': True}]]
        return ins == except_ins


test_func_list = [
    test_max_inputs,
    test_unknown_rank,
    test_unknown_rank_error,
    test_no_intersection,
    test_no_intersection_mul,
    test_const_broadcast_mul,
    test_dim_length_1_mul,
    test_mix_unknown_rank,
    test_empty_shape,
    test_mix_empty_shape,
    test_update_pattern,
    test_mul_input_static
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)
