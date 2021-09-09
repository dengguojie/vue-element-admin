#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from op_test_frame.ut import OpUT
ut_case = OpUT("DynamicGRUV2", "impl.dynamic.dynamic_gru_v2", "dynamic_gru_v2")

# def gen_dynamic_gru_v2_case(shape_x, shape_w, shape_b, shape_output, dtype, init_from_gm, gate_output, with_seq_mask, expect, case_name_val, range_x, range_w, range_b, range_out):
#     if not init_from_gm:
#         init_h = None
#         init_c = None
#     else:
#         shape_init = [1]
#         for i in range(1, len(shape_output)):
#             shape_init.append(shape_output[i])
#         print(shape_init)
#         init_h = {"shape": shape_init, "dtype": "float16", "ori_shape": shape_init, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#         init_c = {"shape": shape_init, "dtype": dtype, "ori_shape": shape_init, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#     if with_seq_mask:
#         seq_mask = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
#                     "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#     else:
#         seq_mask = None
#     if not gate_output:
#         i = None
#         j = None
#         f = None
#         o = None
#         tanhc = None
#     else:
#         i = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#         j = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#         f = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#         o = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}
#         tanhc = {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"}

#     return {"params": [{"shape": shape_x, "dtype": "float16", "ori_shape": shape_x, "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_x},
#                        {"shape": shape_w, "dtype": "float16", "ori_shape": shape_w,
#                         "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM", "range": range_w},
#                        {"shape": shape_b, "dtype": dtype, "ori_shape": shape_b, "ori_format": "ND", "format": "ND", "range": range_b},
#                        seq_mask, init_h, init_c, None, None, None, None,
#                        {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
#                         "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
#                        {"shape": shape_output, "dtype": "float16", "ori_shape": shape_output,
#                         "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
#                        {"shape": shape_output, "dtype": dtype, "ori_shape": shape_output,
#                         "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ", "range": range_out},
#                        i, j, f, o, tanhc],
#             "case_name": case_name_val,
#             "expect": expect,
#             "format_expect": [],
#             "support_expect": True}
def get_params(t_size, m_size, in_x, hidden_size, bias_dtype, data_range=[0.01, 0.1], init_data_range=[0.01, 0.1], seq_dtype='int32'):
    dtype = 'float16'
    shape_w1 = [in_x, 3*hidden_size, 16, 16]
    shape_w2 = [hidden_size, 3*hidden_size, 16, 16]
    if seq_dtype == 'int32':
        shape_seq = [m_size * 16,]
    else:
        shape_seq = [t_size, hidden_size, m_size, 16, 16]
    shape_c = [t_size, hidden_size, m_size, 16, 16]
    shape_c_1 = [1, hidden_size, m_size, 16, 16]
    shape_bias = [3* hidden_size*16,]
    shape_x = [t_size, in_x, m_size, 16, 16]

    x = {"shape":shape_x, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    w1 = {"shape":shape_w1, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    w2 = {"shape":shape_w2, "dtype":dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    seq = {"shape":shape_seq, "dtype":seq_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    b1 = {"shape":shape_bias, "dtype":bias_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    b2 = {"shape":shape_bias, "dtype":bias_dtype, "param_type": "input", "value_range": data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    s_init_h_gm = {"shape":shape_c_1, "dtype":bias_dtype, "param_type": "input", "value_range": init_data_range, "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    output_y = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    output_h = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    i = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    r = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    n = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    hn = {"shape":shape_c, "dtype":bias_dtype, "param_type": "output", "ori_format": "NC1HWC0", "format": "NC1HWC0", "ori_shape": shape_x}
    return x, w1, w2, b1, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn


# case1 = gen_dynamic_gru_v2_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, False,
#                              "success", "dynamic_rnn_1", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
#                              [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])
# case2 = gen_dynamic_gru_v2_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float32", True, True, False,
#                              "success", "dynamic_rnn_2", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
#                              [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])
# case3 = gen_dynamic_gru_v2_case((1,1,2,16,16), (2,4,16,16), (4*16,), (1,1,2,16,16), "float16", True, True, True,
#                              "success", "dynamic_rnn_3", [(1,1),(1,1),(2,2),(16,16),(16,16)], [(2,2),(4,4),(16,16),(16,16)],
#                              [(64,64)], [(1,1),(1,1),(2,2),(16,16),(16,16)])

# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)

# for cov
x, w1, w2, b1, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn = get_params(t_size=2, m_size=1, in_x=64, hidden_size=32, bias_dtype='float32')
ut_case.add_case("all", {
    "params": [x, w1, w2, b1, b2, seq, None, output_y, output_h, None, None, None, None]
})
ut_case.add_case("all", {
    "params": [x, w1, w2, b1, b2, None, None, output_y, output_h, None, None, None, None]
})
ut_case.add_case("all", {
    "params": [x, w1, w2, b1, b2, seq, s_init_h_gm, output_y, output_h, i, r, n, hn]
})
ut_case.add_case("all", {
    "params": [x, w1, w2, b1, b2, None, s_init_h_gm, output_y, output_h, i, r, n, hn]
})


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)


