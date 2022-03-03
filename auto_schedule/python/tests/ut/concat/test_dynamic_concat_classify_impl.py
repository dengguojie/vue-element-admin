# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from tbe.dsl.base.classifier import classify_concat
from tbe.common.context import op_context
from tbe.dsl.base.operation import get_compile_info

warnings.filterwarnings("ignore")
ut_case = OpUT("concat_classify", "concat.test_dynamic_concat_classify_impl")


def test_max_inputs(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 64]
        extra_params = {"axis": 1}
        try:
            ins = classify_concat(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'input numbers error, input numbers must '
                                               'be greater 0 and less equal 63 , now, it is 64'}
            return error_message == e.args[0]
    return False


def test_min_inputs(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 0]
        extra_params = {"axis": 1}
        try:
            ins = classify_concat(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'input numbers error, input numbers must '
                                               'be greater 0 and less equal 63 , now, it is 0'}
            return error_message == e.args[0]
    return False


def test_concat_no_axis_attrs(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 2]
        try:
            ins = classify_concat(input_s, {})
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'inputs of classify must include the dict extra_params '
                                               'with the key axis when mode is concat '}
            return error_message == e.args[0]
    return False


def test_ins_too_much(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 2] * 2
        extra_params = {"axis": 1}
        try:
            ins = classify_concat(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'input numbers error'}
            return error_message == e.args[0]
    return False


def test_unknown_rank_input_too_much(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-2,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 3]
        extra_params = {"axis": 2}
        try:
            ins = classify_concat(input_s, extra_params)
        except RuntimeError as e:
            error_message = {'errCode': 'E90001',
                             'detailed_cause': 'if the shape contains -2, it must be [-2] or (-2,)'}
            return error_message == e.args[0]
    return False


def test_zero_axis_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 4
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 2]
        extra_params = {"axis": 0}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'concat'}, {'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins


def test_one_axis_all_dynamic_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 3]
        extra_params = {"axis": 1}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins


def test_one_axis_part_const_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, 10, 100), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_1 = {"shape": (99, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_2 = {"shape": (-1, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0, input_1, input_2]]
        extra_params = {"axis": 1}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [99, 1000], 'range': [(99, 99), (1, None)], 'mode': 'concat'}, {'shape': [99, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [99, -1], 'range': [(99, 99), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins


def test_one_input_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0]]
        extra_params = {"axis": 1}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins


def test_no_concat_axis_zero_shape_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, 10, 100), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_1 = {"shape": (0, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_2 = {"shape": (-1, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0, input_1, input_2]]
        extra_params = {"axis": 1}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}], 0]]
    return ins == except_ins


def test_concat_axis_zero_shape_concat(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1, 10, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_1 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_2 = {"shape": (-1, 10, 0), "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0, input_1, input_2]]
        extra_params = {"axis": 2}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, 0], 'range': [(1, None), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins


def test_no_concat_axis_zero_range_concat(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None), (1, None), (0, None)]}
        input_1 = {"shape": (-1, -1, 88), "dtype": "float32", "range": [(1, None), (0, None), (88, 88)]}
        input_2 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None), (1, None), (1, None)]}
        input_s = [[input_0, input_1, input_2]]
        extra_params = {"axis": 2}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [-1, -1], 'range': [(1, None), (0, None)], 'mode': 'concat'}, {'shape': [-1, 88], 'range': [(0, None), (88, 88)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}], 1], [[{'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}], 0]]
    return ins == except_ins


def test_concat_axis_zero_range_concat(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None), (1, None), (0, None)]}
        input_1 = {"shape": (-1, 99, -1), "dtype": "float32", "range": [(1, None), (99, 99), (0, 88)]}
        input_2 = {"shape": (-1, -1, -1), "dtype": "float32", "range": [(1, None), (1, None), (0, None)]}
        input_s = [[input_0, input_1, input_2]]
        extra_params = {"axis": 2}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [-1, -1], 'range': [(1, None), (0, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(99, None), (0, 88)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (0, None)], 'mode': 'concat'}], 1], [[{'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}], 0]]
    return ins == except_ins


def test_unknow_rank(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-2,), "dtype": "float32", "range": [(1, None)]}
        input_s = [[input_0] * 3]
        extra_params = {"axis": 2}
        ins = classify_concat(input_s, extra_params)
        except_ins = [[[{'shape': [-1, -1], 'range': [(0, None), (0, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(0, None), (0, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(0, None), (0, None)], 'mode': 'concat'}], 1], [[{'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}, {'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}], 0]]
    return ins == except_ins


def test_one_axis_all_dynamic_concat_neg_axis(_):
    with op_context.OpContext("dynamic"):
        dim_len = 3
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0] * 3]
        extra_params = {"axis": -2}
        ins = classify_concat(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {"_ori_axis": -2}
        import json
        print(json.dumps(compile_info))
        except_ins = [[[{'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}, {'shape': [-1, -1], 'range': [(1, None), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins and compile_info == except_compile_info


def test_one_input_concat_neg_axis(_):
    with op_context.OpContext("dynamic"):
        dim_len = 2
        input_0 = {"shape": (-1,) * dim_len, "dtype": "float32", "range": [(1, None)] * dim_len}
        input_s = [[input_0]]
        extra_params = {"axis": -1}
        ins = classify_concat(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {"_ori_axis": 0}
        except_ins = [[[{'shape': [1, -1], 'range': [(1, 1), (1, None)], 'mode': 'concat'}], 1]]
    return ins == except_ins and compile_info == except_compile_info


def test_unknown_rank_1_axis(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-2,), "dtype": "float32"}
        input_1 = {"shape": (100,), "dtype": "float32", "range": [(100, 100)]}
        input_s = [[input_0, input_1]]
        extra_params = {"axis": 0}
        ins = classify_concat(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {"_ori_axis": 0}
        except_ins = [[[{'shape': [1, -1], 'range': [(1, 1), (0, None)], 'mode': 'concat'}, {'shape': [1, 100], 'range': [(1, 1), (100, 100)], 'mode': 'concat'}], 1]]
    return ins == except_ins and compile_info == except_compile_info


def test_unknown_rank_3_axis(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-2,), "dtype": "float32"}
        input_1 = {"shape": (5, 100, 20), "dtype": "float32", "range": [(5, 5), (100, 100), (20, 20)]}
        input_s = [[input_0, input_1]]
        extra_params = {"axis": -1}
        ins = classify_concat(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {"_ori_axis": -1}
        except_ins = [[[{'shape': [500, -1], 'range': [(500, 500), (0, None)], 'mode': 'concat'}, {'shape': [500, 20], 'range': [(500, 500), (20, 20)], 'mode': 'concat'}], 1]]
    return ins == except_ins and compile_info == except_compile_info


def test_unknown_rank_single_input(_):
    with op_context.OpContext("dynamic"):
        input_0 = {"shape": (-2,), "dtype": "float32"}
        input_s = [[input_0]]
        extra_params = {"axis": -1}
        ins = classify_concat(input_s, extra_params)
        compile_info = get_compile_info()
        except_compile_info = {"_ori_axis": 0}
        except_ins = [[[{'shape': [1, -1], 'range': [(1, 1), (0, None)], 'mode': 'concat'}], 1], [[{'shape': (0,), 'range': [(0, 0)], 'mode': 'concat_empty'}], 0]]
    return ins == except_ins and compile_info == except_compile_info


test_func_list = [
    test_max_inputs,
    test_min_inputs,
    test_concat_no_axis_attrs,
    test_ins_too_much,
    test_unknown_rank_input_too_much,
    test_zero_axis_concat,
    test_one_axis_all_dynamic_concat,
    test_one_axis_part_const_concat,
    test_one_input_concat,
    test_no_concat_axis_zero_shape_concat,
    test_concat_axis_zero_shape_concat,
    test_no_concat_axis_zero_range_concat,
    test_concat_axis_zero_range_concat,
    test_unknow_rank,
    test_one_axis_all_dynamic_concat_neg_axis,
    test_one_input_concat_neg_axis,
    test_unknown_rank_1_axis,
    test_unknown_rank_3_axis,
    test_unknown_rank_single_input
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
