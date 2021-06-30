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
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "ND_RNN_BIAS", "format": "ND_RNN_BIAS"},
                       "ND",
                       "ND_RNN_BIAS",
                       input_size,
                       hidden_size],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def _pad_len(x, y):
    return (y - x % y) % y


def calc_expect_func(src, dst, src_format, dst_format, input_size, hidden_size):
    shape_in = src.get("shape")
    input_tensor = src.get("value")
    output_data = np.array([])
    axis_n = shape_in[0]
    hidden_cnt = axis_n // hidden_size
    for idx in range(hidden_cnt):
        axis_c0 = 16
        c_pad = _pad_len(hidden_size, 16)
        tmp_input_tensor = np.pad(input_tensor[hidden_size * idx:hidden_size * (idx + 1)], ((0, c_pad)),
                                  mode="constant", constant_values=(0, 0))
        output_data = np.hstack((output_data, tmp_input_tensor))
    return output_data


def gen_trans_data_precision_case(src, dst, dtype, input_size, hidden_size, case_name_val, expect):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src, "ori_format": "ND", "format": "ND", "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst, "ori_format": "ND_RNN_BIAS", "format": "ND_RNN_BIAS", "param_type": "output"},
                       "ND",
                       "ND_RNN_BIAS",
                       input_size,
                       hidden_size],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

#ut_case.add_precision_case(["Ascend910", "Ascend310"],
#                           gen_trans_data_precision_case((24,), (32,),
#                                                         "float16", 1, 12, "nd_2_rnn_bias_precision_001",
#                                                         "success"))
ut_case.add_case("Ascend310", gen_trans_data_case((24,), (32,), "float16", 1, 12, "nd_2_rnn_bias_1", "success"))
if __name__ == '__main__':
    simulator_lib_path ="/home/shenmin/Ascend/toolkit/tools/simulator"
    ut_case.run(["Ascend910"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    #ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
