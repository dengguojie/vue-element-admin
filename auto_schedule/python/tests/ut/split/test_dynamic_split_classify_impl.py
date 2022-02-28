# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe.dsl.base.classifier import classify_split
from tbe.common.context import op_context
from tbe.dsl.base.operation import get_compile_info

warnings.filterwarnings("ignore")
ut_case = OpUT("split_classify", "split.test_dynamic_split_classify_impl")


def test_ins_no_num_split(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [input_0, 1]
        extra_params = {"split_num": 2}
        try:
            ins = classify_split(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001', 'detailed_cause':
            'inputs of classify must include the dict extra_params with the key num_split when mode is split'}
            return error_message == e.args[0]
    return False


def test_ins_too_much(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [input_0]
        extra_params = {"num_split": 2}
        try:
            ins = classify_split(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001', 'detailed_cause': 'input numbers error'}
            return error_message == e.args[0]
    return False


def test_unknown_rank_input_too_much(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        axis = 2
        input_0 = {"shape": (-2, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        try:
            ins = classify_split(input_s, extra_params)
        except RuntimeError as e:
            error_message = {
                'errCode': 'E90001',
                'detailed_cause': 'if the shape contains -2, it must be [-2] or (-2,)'
            }
            return error_message == e.args[0]
    return False


def test_zero_axis_split(_):
    with op_context.OpContext("dynamic"):
        dim_len = 4
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = 0
        input_s = [input_0, axis]
        extra_params = {"avg_split": True, "num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins


def test_one_axis_all_dynamic_split(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = 1
        input_s = [input_0, axis]
        extra_params = {"avg_split": True, "num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins


def test_one_axis_part_const_split(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, 10, 100), "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = 1
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [-1, 1000], 'range': [(1, None), (1000, 1000)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, 1000], 'range': [(1, None), (1000, 1000)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins


def test_zero_shape_splt(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (0, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = 2
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': (0, 0), 'range': [(0, 0), (0, 0)], 'mode': 'split_empty', 'split_factor': 1}, 0, [-1, -1]]]
    return ins == except_ins


def test_zero_range_splt(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None), (1, None), (0, None)]}
        axis = 2
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [-1, -1], 'range': [(1, None), (0, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, -1], 'range': [(1, None), (0, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]], [{'shape': (0, 0), 'range': [(0, 0), (0, 0)], 'mode': 'split_empty', 'split_factor': 1}, 0, [-1, -1]]]
    return ins == except_ins


def test_unknow_rank(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-2, ), "dtype": "float32", "range": [(1, None)]}
        axis = 2
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [-1, -1], 'range': [(0, None), (0, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, -1], 'range': [(0, None), (0, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]], [{'shape': (0, 0), 'range': [(0, 0), (0, 0)], 'mode': 'split_empty', 'split_factor': 1}, 0, [-1, -1]]]
    return ins == except_ins


def test_one_axis_all_dynamic_split_neg_axis(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = -2
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {'_ori_axis': -2}
        except_ins = [[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins and compile_info == except_compile_info


def test_single_output_split(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = 1
        input_s = [input_0, axis]
        extra_params = {"avg_split": True, "num_split": 1}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1]]]
    return ins == except_ins


def test_unknow_axis(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, ) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = {"shape": (1, ), "dtype": "float32", "range": [(1, 1)]}
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {'_ori_axis': 1}
        except_ins = [[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins and compile_info == except_compile_info


def test_know_axis(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, 10, 100), "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = {"shape": (1, ), "dtype": "float32", "range": [(1, 1)], "value": (1, )}
        input_s = [input_0, axis]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [-1, 1000], 'range': [(1, None), (1000, 1000)], 'mode': 'split', 'split_factor': 1}, 1, [-1, -1]], [{'shape': [-1, 1000], 'range': [(1, None), (1000, 1000)], 'mode': 'split_general', 'split_factor': 128}, 1, [-1, -1]]]
    return ins == except_ins


def test_const_split_size(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (100, 10, 100), "dtype": "float32", "range": [(1, None)] * dim_len}
        axis = {"shape": (1, ), "dtype": "float32", "range": [(1, 1)], "value": (1, )}
        split_size = {"shape": (2, ), "dtype": "float32", "range": [(1, 1)], "value": (2, 8)}
        input_s = [input_0, axis, split_size]
        extra_params = {"num_split": 2}
        ins = classify_split(input_s, extra_params)
        except_ins = [[{'shape': [100, 1000], 'range': [(100, 100), (1000, 1000)], 'mode': 'split', 'split_factor': 1}, 1, [200, 800]], [{'shape': [100, 1000], 'range': [(100, 100), (1000, 1000)], 'mode': 'split_general', 'split_factor': 128}, 1, [200, 800]]]
    return ins == except_ins

test_func_list = [
    test_ins_no_num_split,
    test_ins_too_much,
    test_unknown_rank_input_too_much,
    test_zero_axis_split,
    test_one_axis_all_dynamic_split,
    test_one_axis_part_const_split,
    test_zero_shape_splt,
    test_zero_range_splt,
    test_unknow_rank,
    test_one_axis_all_dynamic_split_neg_axis,
    test_single_output_split,
    test_unknow_axis,
    test_know_axis,
    test_const_split_size
]

for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
