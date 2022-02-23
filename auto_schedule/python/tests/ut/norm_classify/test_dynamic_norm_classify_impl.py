# # -*- coding:utf-8 -*-
import warnings
from sch_test_frame.ut import OpUT
import tbe
from tbe.dsl.base.classifier import classify_norm
from tbe.dsl.base.operation import get_compile_info

warnings.filterwarnings("ignore")
ut_case = OpUT("norm_classify", "norm_classify.test_dynamic_norm_classify_impl")


def test_norm_classify_0(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float16", "shape": (-2, )}, {"dtype": "float16", "shape": (-2, )},
            {"dtype": "float16", "shape": (-2, )}, {"dtype": "float16", "shape": (-2, )},
            {"dtype": "float16", "shape": (-2, )}, 'unknown'
        ]
        extra_params = {
            "input_shape_type": [0, 1, 1, 1, 1], "same_input_shape_group": [[1, 2], [3, 4]],
            "reduce_axes_type": "after",
            "broadcast_axes_type": {1: "same_reduce", 2: "same_reduce", 3: "opposite_reduce", 4: "opposite_reduce"},
            "compile_broadcast_axes": {1: "unknown", 2: "unknown", 3: "unknown", 4: "unknown"}
        }
        ins = classify_norm(inputs, extra_params)

        expect_ins = [
            [
                {'shape': [-1], 'range': [(1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 2010},
                {'shape': [1], 'range': [(1, 1)], 'mode': 'broadcast_reduce_equal', 'input_type': 1,
                 'broadcast_axis': [0], 'norm_pattern': 2010},
                {'shape': [1], 'range': [(1, 1)], 'mode': 'broadcast_reduce_equal', 'input_type': 1,
                 'broadcast_axis': [0], 'norm_pattern': 2010},
                {'shape': [-1], 'range': [(1, None)], 'mode': 'no_broadcast', 'input_type': 2,
                 'broadcast_axis': [], 'norm_pattern': 2010},
                {'shape': [-1], 'range': [(1, None)], 'mode': 'no_broadcast', 'input_type': 2,
                 'broadcast_axis': [], 'norm_pattern': 2010},
                [0]
            ],
            [
                {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 4012},
                {'shape': [-1, 1], 'range': [(1, None), (1, 1)], 'mode': 'broadcast_reduce_equal', 'input_type': 1,
                 'broadcast_axis': [1], 'norm_pattern': 4012},
                {'shape': [-1, 1], 'range': [(1, None), (1, 1)], 'mode': 'broadcast_reduce_equal', 'input_type': 1,
                 'broadcast_axis': [1], 'norm_pattern': 4012},
                {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'broadcast_reduce_opposite', 'input_type': 2,
                 'broadcast_axis': [0], 'norm_pattern': 4012},
                {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'broadcast_reduce_opposite', 'input_type': 2,
                 'broadcast_axis': [0], 'norm_pattern': 4012},
                [1]
            ],
        ]
        compile_info = get_compile_info()
        is_true = \
            compile_info["_input_type"] == [0, 1, 1, 2, 2] and compile_info["_reduce_axis_type"] == 3 and\
            compile_info["_broadcast_axis_type_list"] == [1, 2]

        return ins == expect_ins and is_true


def test_norm_classify_1(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float16", "shape": (-2, )}, {"dtype": "float16", "shape": (-2, )},
            {"dtype": "float16", "shape": (-2, )}, 'unknown'
        ]
        extra_params = {
            "input_shape_type": [0, 1, 1], "same_input_shape_group": [[1, 2]],
            "reduce_axes_type": "after",
            "broadcast_axes_type": {1: "opposite_reduce", 2: "opposite_reduce",},
            "compile_broadcast_axes": {1: "unknown", 2: "unknown",}
        }
        ins = classify_norm(inputs, extra_params)

        expect_ins = [
            [
                {'shape': [-1], 'range': [(1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 2001},
                {'shape': [-1], 'range': [(1, None)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [], 'norm_pattern': 2001},
                {'shape': [-1], 'range': [(1, None)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [], 'norm_pattern': 2001},
                [0]
            ],
            [
                {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 4005},
                {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [0], 'norm_pattern': 4005},
                {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [0], 'norm_pattern': 4005},
                [1]
            ],
            [
                {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 5006},
                {'shape': [1, 1], 'range': [(1, 1), (1, 1)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [0, 1], 'norm_pattern': 5006},
                {'shape': [1, 1], 'range': [(1, 1), (1, 1)], 'mode': 'broadcast_axis_known', 'input_type': 1,
                 'broadcast_axis': [0, 1], 'norm_pattern': 5006},
                [0]
            ],
        ]
        compile_info = get_compile_info()
        is_true = \
            compile_info["_input_type"] == [0, 1, 1] and compile_info["_reduce_axis_type"] == 3 and \
            compile_info["_broadcast_axis_type_list"] == [2] and "_ori_broadcast_axis" in compile_info

        return ins == expect_ins and is_true


def test_norm_classify_2(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
        inputs = [
            {"dtype": "float16", "shape": (-2, )},
            'unknown'
        ]
        extra_params = {
            "reduce_axes_type": "single",
        }
        ins = classify_norm(inputs, extra_params)

        expect_ins = [
            [
                {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 5000},
                [0]
            ],
            [
                {'shape': [-1], 'range': [(1, None)], 'mode': 'common', 'input_type': 0,
                 'broadcast_axis': None, 'norm_pattern': 2000},
                [0]
            ],
            [
                {'shape': [-1, -1, -1], 'range': [(1, None), (1, None), (1, None)], 'mode': 'common',
                 'input_type': 0, 'broadcast_axis': None, 'norm_pattern': 9000},
                [1]
            ],
            [
                {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'common',
                 'input_type': 0, 'broadcast_axis': None, 'norm_pattern': 4000},
                [1]
            ],
        ]
        compile_info = get_compile_info()
        is_true = compile_info["_input_type"] == [0] and compile_info["_reduce_axis_type"] == 9

        return ins == expect_ins and is_true


ut_case.add_cust_test_func(test_func=test_norm_classify_0)
ut_case.add_cust_test_func(test_func=test_norm_classify_1)
ut_case.add_cust_test_func(test_func=test_norm_classify_2)
