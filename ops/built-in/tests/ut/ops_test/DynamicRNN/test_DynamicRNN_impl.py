#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
import numpy as np
ut_case = OpUT("DynamicRNN", "impl.dynamic_rnn", "dynamic_rnn")

def gen_dynamic_rnn_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, expect, case_name_val):
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
                       i, j, f, o, tanhc],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_dynamic_rnn_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                             "success", "dynamic_rnn_1")

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)


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


