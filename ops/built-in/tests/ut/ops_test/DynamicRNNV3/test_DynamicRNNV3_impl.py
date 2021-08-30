#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicRNNV3", "impl.dynamic_rnn_v3", "dynamic_rnn_v3")

def gen_dynamic_rnn_v3_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, expect, case_name_val):

    shape_x = [1, 64, 1, 16, 16]
    shape_w = [96, 128, 16, 16]
    shape_b = [2048]
    init_h = {"shape": [1, 2, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 2, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

    init_c = {"shape": [1, 32, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 32, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

    wci = {"shape": [1, 32, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 32, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    wcf = {"shape": [1, 32, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 32, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    wco = {"shape": [1, 32, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 32, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    real_mask = {"shape": [1, 1, 1, 16, 16], "dtype": "float16", "ori_shape": [1, 1, 1, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    project = {"shape": [1, 2, 32, 16, 16], "dtype": "float16", "ori_shape": [1, 2, 32, 16, 16], "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

    shape_output = [1, 32, 2, 16, 16]
    shape_outout2 = [1, 2, 2, 16, 16]
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
    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND", "format": "ND"},
                       None, init_h, init_c, wci, wcf, wco, None, real_mask, project,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_outout2, "dtype": "float16", "ori_shape": shape_outout2,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                       i, j, f, o, tanhc],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_dynamic_rnn_v3_case((1,64,2,16,16), (96,128,16,16), (128*16,), (1,32,2,16,16), "float16", True, True,
                                "success", "dynamic_rnn_v3")

ut_case.add_case(["Ascend310","Ascend710","Ascend910A"], case1)


if __name__ == '__main__':
    ut_case.run("Ascend310")
    exit(0)


