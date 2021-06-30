#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import math
import time
from op_test_frame.common import precision_info

ut_case = OpUT("TransDataRNN", "impl.trans_data_rnn", "trans_data_rnn")


def gen_trans_data_case(src, dst, dtype, input_size, hidden_size, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "ND", "format": "ND"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "FRACTAL_ZN_RNN", "format": "FRACTAL_ZN_RNN"},
                       "ND",
                       "FRACTAL_ZN_RNN",
                       input_size,
                       hidden_size],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def _pad_len(x, y):
    return (y - x % y) % y

def _ceil_div(x, y):
    return (x + y -1) // y

def calc_expect_func(src, dst, src_format, dst_format, input_size, hidden_size):
    shape_in = src.get("shape")
    input_tensor = src.get("value")
    axis_c, axis_n = shape_in[0], shape_in[1]
    axis_c0 = 16
    axis_ni = 16
    axis_no = _ceil_div(hidden_size, axis_ni)
    n_pad = _pad_len(hidden_size, 16)
    c_pad = _pad_len(axis_c, axis_c0)
    axis_c1 = _ceil_div(axis_c, axis_c0)
    hidden_cnt = axis_n // hidden_size
    tmp_input_tensor = np.pad(input_tensor[:, 0 : hidden_size], ((0, c_pad), (0, n_pad)),
                              mode="constant", constant_values=(0, 0))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_c1, axis_c0, axis_no, axis_ni)
    data_y = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1))
    output_data = data_y
    for idx in range(1, hidden_cnt):
        tmp_input_tensor = np.pad(input_tensor[:, hidden_size*idx : hidden_size*(idx+1)], ((0, c_pad), (0, n_pad)),
                                mode="constant", constant_values=(0, 0))
        tmp_input_tensor = tmp_input_tensor.reshape(axis_c1, axis_c0, axis_no, axis_ni)
        data_y = np.transpose(tmp_input_tensor, axes=(0, 2, 3, 1))
        output_data = np.hstack((output_data, data_y))
    return output_data


def gen_trans_data_precision_case(src, dst, dtype, input_size, hidden_size, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "ND", "format": "ND", "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "FRACTAL_ZN_RNN", "format": "FRACTAL_ZN_RNN", "param_type": "output"},
                       "ND",
                       "FRACTAL_ZN_RNN",
                       input_size,
                       hidden_size],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((2, 24), (1, 2, 16, 16),
                                                         "float16", 2, 12, "nd_2_fractal_rnn_precision_001",
                                                         "success"))
ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((96, 288), (6, 20, 16, 16),
                                                         "float16", 96, 72, "nd_2_fractal_rnn_precision_001",
                                                         "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((4000,8000), (250, 500, 16, 16), "float16", 4000, 4000, "nd2fractal_rnn_1", "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((17,4000), (2, 250, 16, 16), "float16", 17, 4000, "nd2fractal_rnn_1", "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((2,3900), (1, 244, 16, 16), "float16", 2, 3900, "nd2fractal_rnn_1", "success"))


if __name__ == '__main__':
    simulator_lib_path ="/home/shenmin/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
