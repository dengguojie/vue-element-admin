#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicRNNV2", "impl.dynamic.dynamic_rnn_v2", "dynamic_rnn_v2")

def gen_dynamic_rnn_case(shape_x, shape_wi, shape_wh, shape_b, shape_output, dtype, init_from_gm, gate_output,
                         activation, recurrent_activation, gate_order, with_seq_mask, expect, case_name_val):
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
        seq_mask = {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
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

    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_wi, "dtype": "float16", "ori_shape": shape_wi,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": shape_wh, "dtype": "float16", "ori_shape": shape_wh,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND", "format": "ND"},
                       seq_mask, init_h, init_c, None, None, None, None,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       i, j, f, o, tanhc, "LSTM", "UNIDIRECTIONAL", 1, False, 1.0, -1.0, 0, True,
                       activation, recurrent_activation, 0.0, gate_order, False, "concat", True],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True
            }

case1 = gen_dynamic_rnn_case((1,2,1,16,16), (2,16,16,16), (4,16,16,16), (16*16,), (1,4,1,16,16), "float16", True, True,
                             "tanh", "sigmoid", "ijfo", False, "success", "dynamic_rnn_1")
case2 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "clip", "hard_sigmoid", "ifco", False, "success", "dynamic_rnn_2")
case3 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float32", True, True,
                             "tanh", "sigmoid", "ijfo", False, "success", "dynamic_rnn_3")
case4 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float32", True, True,
                             "clip", "hard_sigmoid", "ifco", False, "success", "dynamic_rnn_4")
case5 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float32", True, True,
                             "clip", "hard_sigmoid", "ijfo", True, "success", "dynamic_rnn_5")
case6 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float16", False, True,
                             "tanh", "hard_sigmoid", "ijfo", True, "success", "dynamic_rnn_6")
case7 = gen_dynamic_rnn_case((1,64,2,16,16), (64,4*32,16,16), (32,32*4,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "tanh", "hard_sigmoid", "ijfo", True, "success", "dynamic_rnn_7")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)