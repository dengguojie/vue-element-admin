#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicRNNV3", "impl.dynamic.dynamic_rnn_v3", "dynamic_rnn_v3")

def gen_dynamic_rnnv3_case(shape_x, shape_w, shape_b, dtype, init_from_gm, gate_output, with_seq_mask, cell_clip,
                           wc_exit, real_mask_exit, project_exit,
                           expect, case_name_val, range_x, range_w, range_b):
    t = shape_x[0]
    x = shape_x[1]
    m = shape_x[2]
    state = shape_w[0] - x
    hidden = shape_w[1] // 4
    shape_output = [t, state, m, 16, 16]
    shape_output_c = [t, hidden, m, 16, 16]
    range_out = [range_x[0], (state, state), range_x[2], (16,16), (16,16)]
    range_out_c = [range_x[0], (hidden, hidden), range_x[2], (16,16), (16,16)]
    if not wc_exit:
        wci = None
        wcf = None
        wco = None
    else:
        shape_wc = [hidden, m, 16, 16]
        wci = {"shape": shape_wc, "dtype": dtype, "ori_shape": shape_wc, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        wcf = {"shape": shape_wc, "dtype": dtype, "ori_shape": shape_wc, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        wco = {"shape": shape_wc, "dtype": dtype, "ori_shape": shape_wc, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    if not real_mask_exit:
        real_mask = None
    else:
        shape_mask = [t, 1, m, 16, 16]
        real_mask = {"shape": shape_mask, "dtype": dtype, "ori_shape": shape_mask, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    if not project_exit:
        project = None
    else:
        shape_project = [state, hidden, 16, 16]
        project = {"shape": shape_project, "dtype": "float16", "ori_shape": shape_project, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    if not init_from_gm:
        init_h = None
        init_c = None
    else:
        shape_init_h = []
        shape_init_c = []
        for i in range(1, len(shape_output)):
            shape_init_h.append(shape_output[i])
            shape_init_c.append(shape_output_c[i])
        init_h = {"shape": shape_init_h, "dtype": "float16", "ori_shape": shape_init_h, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        init_c = {"shape": shape_init_c, "dtype": dtype, "ori_shape": shape_init_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
    if with_seq_mask:
        seq_mask = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c,
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
        i = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        j = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        f = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        o = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
        tanhc = {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

    return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_x},
                       {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
                        "ori_format": "FRACTAL_ZN_RNN", "format": "FRACTAL_ZN_RNN", "range": range_w},
                       {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND_RNN_BIAS", "format": "ND_RNN_BIAS", "range": range_b},
                       seq_mask, init_h, init_c, wci, wcf, wco, None, real_mask, project,
                       {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
                       {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
                       {"shape": shape_output_c, "dtype": dtype, "ori_shape": shape_output_c,
                        "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out_c},
                       i, j, f, o, tanhc, "LSTM", "UNIDIRECTIONAL", 1, False, 1.0, cell_clip, 0, True, "tanh", 0.0, True],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

case1 = gen_dynamic_rnnv3_case((1,1,2,16,16), (2,4,16,16), (4*16,), "float16", True, True, False, -1.0,
                              True, True, True,
                             "success", "dynamic_dynamic_rnnv3_1", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)])
case2 = gen_dynamic_rnnv3_case((1,1,2,16,16), (2,4,16,16), (4*16,), "float16", True, True, False, -1.0,
                              True, True, False,
                             "success", "dynamic_dynamic_rnnv3_2", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)])
case3 = gen_dynamic_rnnv3_case((1,1,2,16,16), (2,4,16,16), (4*16,), "float32", True, True, False, -1.0,
                              True, True, True,
                             "success", "dynamic_dynamic_rnnv3_3", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)])
case4 = gen_dynamic_rnnv3_case((1,1,2,16,16), (2,4,16,16), (4*16,), "float32", True, True, False, -1.0,
                              True, True, False,
                             "success", "dynamic_dynamic_rnnv3_4", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
                             [(64,64)])

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
