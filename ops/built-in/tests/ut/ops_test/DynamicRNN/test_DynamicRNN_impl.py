#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicRNN", "impl.dynamic_rnn", "dynamic_rnn")

def gen_dynamic_rnn_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, expect,
                         direction, gate_order, case_name_val, with_seq_mask):
    if not init_from_gm:
        init_h = None
        init_c = None
    else:
        shape_init = [1]
        for i in range(1, len(shape_output)):
            shape_init.append(shape_output[i])

        init_h = {"shape": shape_init, "dtype": "float16", "ori_shape": shape_init,
                  "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        init_c = {"shape": shape_init, "dtype": dtype, "ori_shape": shape_init,
                  "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
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
        i = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
             "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        j = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
             "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        f = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
             "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        o = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
             "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        tanhc = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                 "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    
    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b,
                        "ori_format": "ND", "format": "ND"},
                       seq_mask, init_h, init_c, None, None, None, None,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       i, j, f, o, tanhc, "LSTM", direction, 1, False, 1.0, -1.0, 0, True, "tanh", 0.0, gate_order],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}
            
case1 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "success", "UNIDIRECTIONAL","ijfo", "dynamic_rnn_1", False)
case2 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                                            "success", "UNIDIRECTIONAL", "ijfo", "dynamic_rnn_2", True)
case3 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "success", "REDIRECTIONAL", "ijfo", "dynamic_rnn_3", False)
case4 = gen_dynamic_rnn_case((1,64,2,16,16), (64,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             RuntimeError, "UNIDIRECTIONAL", "ijfo", "dynamic_rnn_failed_w", False)
case5 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (96*16,), (1,32,2,16,16), "float16", True, True,
                             RuntimeError, "UNIDIRECTIONAL", "ijfo", "dynamic_rnn_failed_b", False)
case5 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (96*16,), (1,32,2,16,16), "float16", True, True,
                             RuntimeError, "UNIDIRECTIONAL", "ijfo", "dynamic_rnn_failed_b", False)
case6 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "success", "REDIRECTIONAL", "ifjo", "dynamic_rnn_6", False)
case7 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             RuntimeError, "UNIDIRECTIONAL", "ifco", "dynamic_rnn_failed_7", False)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
