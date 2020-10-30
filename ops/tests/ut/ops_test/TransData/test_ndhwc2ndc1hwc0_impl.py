#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, dst_dtype, case_name_val, expect,
                        dst_format="NDC1HWC0", src_format="NDHWC"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NDHWC", "format": "NDHWC"},
                       {"shape": dst, "dtype": dst_dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def _ceil_div(x, y):
    res = (x + y - 1) // y
    return res


def _pad_len(x, y):
    res = (y - x % y) % y
    return res


def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    axis_n = input_shape[0]
    axis_d = input_shape[1]
    axis_h = input_shape[2]
    axis_w = input_shape[3]
    axis_c = input_shape[4]
    axis_c0 = 16
    axis_c1 = _ceil_div(axis_c, axis_c0)
    c_pad = _pad_len(axis_c, axis_c0)

    tmp_input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (0, 0), (0, c_pad)),
                              mode="constant", constant_values=(0, 0))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_n, axis_d,
                                                axis_h, axis_w, axis_c1, axis_c0)
    output_tensor = np.transpose(tmp_input_tensor, axes=(0, 1, 4, 2, 3, 5))

    return output_tensor


def gen_trans_data_precision_case(src, dst, dtype, dst_dtype, case_name_val, expect,
                                  dst_format="NDC1HWC0", src_format="NDHWC"):
    c0_len = 32 if dtype == "int8" else 16
    dst = (src[0], src[1], _ceil_div(src[4], c0_len), src[2], src[3], c0_len)

    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NDHWC", "format": "NDHWC",
                        "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dst_dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format, "param_type": "output"},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


# normal
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 16),
                                     "float16", "float16", "ndhwc_1",
                                     "success"))

# exception
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 16),
                                     "float16", "float32", "ndhwc_2",
                                     RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 15),
                                     "float16", "float16", "ndhwc_3",
                                     RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 2, 1, 1, 16),
                                     "float16", "float16", "ndhwc_4",
                                     RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 16),
                                     "float16", "float16", "ndhwc_5",
                                     RuntimeError,
                                     src_format="NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 16),
                                     "float16", "float16", "ndhwc_6",
                                     RuntimeError,
                                     dst_format="NC1HWC0"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 1, 16), (1, 1, 1, 2, 1, 16),
                                     "float16", "float16", "ndhwc_7",
                                     RuntimeError))

# add precision case
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1, 1, 1, 1, 16), (1, 1, 1, 1, 1, 16),
                                                         "float16", "float16", "ndhwc_1",
                                                         "success"))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
