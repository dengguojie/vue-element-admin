#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicRNN", "impl.dynamic.dynamic_rnn", "dynamic_rnn")

def gen_dynamic_rnn_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, with_seq_mask, expect, case_name_val, range_x, range_w, range_b, range_out):
    if not init_from_gm:
        init_h = None
        init_c = None
    else:
        shape_init = [1]
        for i in range(1, len(shape_output)):
            shape_init.append(shape_output[i])
        print(shape_init)
        init_h = {"shape": shape_init, "dtype": "float16", "ori_shape": shape_init, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        init_c = {"shape": shape_init, "dtype": dtype, "ori_shape": shape_init, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    if with_seq_mask:
        seq_mask = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                    "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    else:
        seq_mask = None
    if not gate_output:
        i = None
        j = None
        f = None
        o = None
        tanhc = None
    else:
        i = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        j = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        f = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        o = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        tanhc = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_x},
                       {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM", "range": range_w},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND", "format": "ND", "range": range_b},
                       seq_mask, init_h, init_c, None, None, None, None,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
                       {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
                       i, j, f, o, tanhc],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False,
                             "success", "dynamic_rnn_1", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])
case2 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float32", True, True, False,
                             "success", "dynamic_rnn_2", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])
case3 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, True,
                             "success", "dynamic_rnn_3", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)

from impl.dynamic.dynamic_rnn import dynamic_rnn_np
from impl.dynamic.dynamic_rnn import dynamic_rnn_generalization

x_data = np.ones([1,1,2,16,16], dtype = np.float16)
w_data = np.ones([2,4,16,16], dtype = np.float16)
bias_num = np.ones([4*16], dtype = np.float16)
input_data_list = [x_data, w_data, bias_num]

def test_check_support(test_arg):
    dynamic_rnn_np(input_data_list,
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16,16), "dtype": "float16", "ori_shape": (2,4,16,16), "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                    {"shape": (4*16,), "dtype": "float16", "ori_shape": (4*16,), "ori_format": "ND", "format": "ND"},
                    None, None, None, None, None, None, None,None,
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    None, None, None, None, None)

def test_rnn_generalization(test_arg):
    dynamic_rnn_generalization(
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (32,64), "dtype": "float16", "ori_shape": (32,64), "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                    {"shape": (4*16,), "dtype": "float16", "ori_shape": (4*16,), "ori_format": "ND", "format": "ND"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,4,16), "dtype": "float16", "ori_shape": (1,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    None, None, None, None,
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"})


ut_case.add_cust_test_func(test_func=test_check_support)
ut_case.add_cust_test_func(test_func=test_rnn_generalization)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)


