#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
ut_case = OpUT("DynamicRNN", "impl.dynamic_rnn", "dynamic_rnn")

def gen_dynamic_rnn_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, with_seq_mask, cell_clip, expect, case_name_val):
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

    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND", "format": "ND"},
                       None, init_h, init_c, None, None, None, None,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       i, j, f, o, tanhc, "LSTM", "UNIDIRECTIONAL", 1, False, 1.0, cell_clip, 0, True, "tanh", 0.0, "ijfo", True],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True, False, -1.0,
                             "success", "dynamic_rnn_1")
case2 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float32", True, True, False, -1.0,
                             "success", "dynamic_rnn_2")
case3 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True, False, 1.0,
                             "success", "dynamic_rnn_3")
case4 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float32", True, True, False, 1.0,
                             "success", "dynamic_rnn_4")
case5 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_5")
case5['params'][10]['dtype'] = 'float32'
case6 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_6")
case6['params'][12]['dtype'] = 'float32'
case7 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_7")
case7['params'][13]['dtype'] = 'float32'
case8 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_8")
case8['params'][14]['dtype'] = 'float32'
case9 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_9")
case9['params'][15]['dtype'] = 'float32'
case10 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_10")
case10['params'][16]['dtype'] = 'float32'
case11 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_11")
case11['params'][17]['dtype'] = 'float32'
case12 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,4,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_12")
case13 = gen_dynamic_rnn_case((1,1,2,16,16), (3,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_13")
case14 = gen_dynamic_rnn_case((1,1,2,16,16), (2,5,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_14")
case15 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (5*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_15")
case16 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_16")
case16['params'][4] = None
case17 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_17")
case17['params'][11]['shape'] = (1,1,2,16,4)
case18 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_18")
case18['params'][13]['shape'] = (1,1,2,16,4)
case19 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_19")
case19['params'][14]['shape'] = (1,1,2,16,4)
case20 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_20")
case20['params'][15]['shape'] = (1,1,2,16,4)
case21 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_21")
case21['params'][16]['shape'] = (1,1,2,16,4)
case22 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_22")
case22['params'][17]['shape'] = (1,1,2,16,4)
case23 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_23")
case23['params'][10]['shape'] = (1,1,2,16,4)
case24 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_24")
case24['params'][6] = {"shape": (1,2), "dtype": "float16", "ori_shape": (1,2), 
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
case25 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_25")
case25['params'][7] = {"shape": (1,2), "dtype": "float16", "ori_shape": (1,2), 
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
case26 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_26")
case26['params'][8] = {"shape": (1,2), "dtype": "float16", "ori_shape": (1,2), 
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
case27 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_27")
case27['params'][9] = {"shape": (1,2), "dtype": "float16", "ori_shape": (1,2), 
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
case28 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_28")
case28['params'][18] = "RNN"
case29 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_29")
case29['params'][19] =  "UNIDIREC"
case30 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_30")
case30['params'][20] = 0
case31 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_31")
case31['params'][21] = True
case32 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_32")
case32['params'][22] = 0
case33 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_33")
case33['params'][23] = -2.0
case34 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_34")
case34['params'][24] = 1
case35 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_35")
case35['params'][25] = False
case36 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_36")
case36['params'][26] = "tan"
case37 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_37")
case37['params'][28] = "fo"
case38 = gen_dynamic_rnn_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False, -1.0,
                             RuntimeError, "dynamic_rnn_38")
case38['params'][5]['dtype'] = 'float32'

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case22)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case23)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case24)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case25)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case26)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case27)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case28)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case29)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case30)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case31)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case32)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case33)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case34)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case35)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case36)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case37)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case38)

x_data = np.ones([1,1,2,16,16], dtype = np.float16)
w_data = np.ones([2,4,16,16], dtype = np.float16)
bias_num = np.ones([4*16], dtype = np.float16)
input_data_list = [x_data, w_data, bias_num]

def test_dynamic_rnn_np_000(test_arg):
    from impl.dynamic_rnn import dynamic_rnn_np
    dynamic_rnn_np(input_data_list,
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16,16), "dtype": "float16", "ori_shape": (2,4,16,16), "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                    {"shape": (4*16,), "dtype": "float16", "ori_shape": (4*16,), "ori_format": "ND", "format": "ND"},
                    None, None, None, None, None, None, None,None,
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,1,2,16,16), "dtype": "float16", "ori_shape": (1,1,2,16,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    None, None, None, None, None)

def test_matrix_to_zz(test_arg):
    from impl.dynamic_rnn import matrix_to_zz
    matrix_to_zz(np.ones((1, 32, 16),dtype=np.float16),(1, 32, 16),"float16")
    matrix_to_zz(np.ones((32, 16),dtype=np.float16),(32, 16),"float16")
    matrix_to_zz(np.ones((1, 1, 16),dtype=np.float16),(1, 1, 16),"float16")
    matrix_to_zz(np.ones((1, 16),dtype=np.float16),(1, 16),"float16")

def test_matrix_to_nZ(test_arg):
    from impl.dynamic_rnn import matrix_to_nZ
    matrix_to_nZ(np.ones((1, 32, 16),dtype=np.float16),(1, 32, 16),"float16")
    matrix_to_nZ(np.ones((32, 16),dtype=np.float16),(32, 16),"float16")
    matrix_to_nZ(np.ones((1, 32, 1),dtype=np.float16),(1, 32, 1),"float16")
    matrix_to_nZ(np.ones((32, 1),dtype=np.float16),(32, 1),"float16")

def test_matrix_to_zn(test_arg):
    from impl.dynamic_rnn import matrix_to_zn
    matrix_to_zn(np.ones((1, 32, 16),dtype=np.float16),(1, 32, 16),"float16")
    matrix_to_zn(np.ones((32, 16),dtype=np.float16),(32, 16),"float16")
    matrix_to_zn(np.ones((1, 32, 1),dtype=np.float16),(1, 32, 1),"float16")
    matrix_to_zn(np.ones((32, 1),dtype=np.float16),(32, 1),"float16")
    matrix_to_zn(np.ones((1, 1, 16),dtype=np.float16),(1, 1, 16),"float16")
    matrix_to_zn(np.ones((1, 16),dtype=np.float16),(1, 16),"float16")

def test_maxtrix_zn_reverse(test_arg):
    from impl.dynamic_rnn import maxtrix_zn_reverse
    maxtrix_zn_reverse(np.ones((512,),dtype=np.float16),((1, 2, 16, 16)),"float16")

def test_maxtrix_nz_reverse(test_arg):
    from impl.dynamic_rnn import maxtrix_nz_reverse
    maxtrix_nz_reverse(np.ones((512,),dtype=np.float16),((1, 1, 2, 16, 16)),"float16")

ut_case.add_cust_test_func(test_func=test_dynamic_rnn_np_000)
ut_case.add_cust_test_func(test_func=test_matrix_to_zz)
ut_case.add_cust_test_func(test_func=test_matrix_to_nZ)
ut_case.add_cust_test_func(test_func=test_matrix_to_zn)
ut_case.add_cust_test_func(test_func=test_maxtrix_zn_reverse)
ut_case.add_cust_test_func(test_func=test_maxtrix_nz_reverse)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
