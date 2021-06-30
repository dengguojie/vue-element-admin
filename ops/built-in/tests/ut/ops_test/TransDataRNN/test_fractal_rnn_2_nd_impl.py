#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
import math
import time
from op_test_frame.common import precision_info

ut_case = OpUT("TransDataRNN", "impl.trans_data_rnn", "trans_data_rnn")


def gen_trans_data_case(src, dst, dtype, input_size, hidden_size, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "FRACTAL_ZN_RNN", "format": "FRACTAL_ZN_RNN"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "ND", "format": "ND"},
                       "FRACTAL_ZN_RNN",
                       "ND",
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
    shape_out = dst.get("shape")
    input_tensor = src.get("value")
    axis_c, axis_n = shape_out[0], shape_out[1]
    axis_c0 = 16
    axis_ni = 16
    axis_no = shape_in[1]
    axis_c1 = shape_in[0]
    n_pad = _pad_len(hidden_size, 16)
    c_pad = None if axis_c1 * axis_c0 == input_size else input_size - axis_c1 * axis_c0
    hidden_cnt = axis_n // hidden_size
    offset = hidden_size + n_pad
    offset_n = (offset // 16)
    tmp_input_tensor = input_tensor[:, :offset_n, :, :]
    tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_c1 * axis_c0, offset)
    data_y = tmp_input_tensor[:c_pad, :hidden_size]
    output_data = data_y
    for idx in range(1, hidden_cnt):
        tmp_input_tensor = input_tensor[:, offset_n*idx:offset_n*(idx + 1), :, :]
        tmp_input_tensor = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2))
        tmp_input_tensor = tmp_input_tensor.reshape(axis_c1 * axis_c0, offset)
        data_y = tmp_input_tensor[:c_pad, :hidden_size]
        output_data = np.hstack((output_data, data_y))
    return output_data

def gen_trans_data_precision_case(src, dst, dtype, input_size, hidden_size, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "FRACTAL_ZN_RNN", "format": "FRACTAL_ZN_RNN", "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "ND", "format": "ND", "param_type": "output"},
                       "FRACTAL_ZN_RNN",
                       "ND",
                       input_size,
                       hidden_size],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((1, 2, 16, 16), (2, 24), 
                                                         "float16", 2, 12, "fractal_rnn_2_nd_precision_001",
                                                         "success"))
ut_case.add_precision_case(["Ascend910", "Ascend310"],
                           gen_trans_data_precision_case((6, 20, 16, 16), (96, 288), 
                                                         "float16", 96, 72, "fractal_rnn_2_nd_precision_002",
                                                         "failed"))
ut_case.add_case("Ascend310", gen_trans_data_case((250, 500, 16, 16), (4000,8000), "float16", 4000, 4000, "fractal_rnn2nd_1", "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((1, 500, 16, 16), (2,8000), "float16", 2, 2000, "fractal_rnn2nd_1", "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((1, 5, 16, 16), (2,80), "float16", 2, 2000, "fractal_rnn2nd_1", "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((2, 500, 16, 16), (17,80), "float16", 17, 2000, "fractal_rnn2nd_1", "success"))

if __name__ == '__main__':
    simulator_lib_path ="/home/shenmin/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
