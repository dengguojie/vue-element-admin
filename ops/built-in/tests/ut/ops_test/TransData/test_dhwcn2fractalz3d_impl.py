#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data",
               "trans_data")


def _ceil_div(x, y):
    res = (x + y - 1) // y
    return res


def _pad_len(x, y):
    res = (y - x % y) % y
    return res


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z_3D"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "DHWCN", "format": "DHWCN"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       "DHWCN", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def gen_trans_data_case_hwcn(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "HWCN", "format": "HWCN"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       "HWCN", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    axis_d = input_shape[0]
    axis_h = input_shape[1]
    axis_w = input_shape[2]
    axis_c = input_shape[3]
    axis_n = input_shape[4]
    axis_ni = 16
    axis_c0 = 16
    axis_c1 = _ceil_div(axis_c, axis_c0)
    axis_no = _ceil_div(axis_n, axis_ni)
    c_pad = _pad_len(axis_c, axis_c0)
    n_pad = _pad_len(axis_n, axis_ni)

    tmp_input_tensor = np.pad(input_tensor, ((0, 0), (0, 0), (0, 0), (0, c_pad), (0, n_pad)),
                              mode="constant", constant_values=(0, 0))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_d, axis_h, axis_w,
                                                axis_c1, axis_c0, axis_no, axis_ni)
    output_tensor = np.transpose(tmp_input_tensor, axes=(0, 3, 1, 2, 5, 6, 4))
    output_shape = dst.get("shape")
    output_tensor = output_tensor.reshape(output_shape)

    return output_tensor


def gen_trans_data_precision_case(src, dst, dtype, case_name_val, expect,
                                  dst_format="FRACTAL_Z_3D"):
    c0_len = 32 if dtype == "int8" else 16
    dst = (src[0] * _ceil_div(src[3], c0_len)* src[1]* src[2], _ceil_div(src[4], 16), 16, c0_len)

    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "DHWCN", "format": "DHWCN", "param_type": "input",
                        "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format, "param_type": "output"},
                       "DHWCN", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


# c < 16 and n == 1
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 12, 3, 1), (3, 2, 16, 16),
                                     "float16", "dhwcn_1", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 12, 16, 1), (3, 2, 16, 16),
                                     "float16", "dhwcn_2", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3600, 96, 12, 1, 1), (3, 2, 16, 16),
                                     "float16", "dhwcn_3", "success"))

ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case_hwcn((6, 12, 3, 2), (72, 1, 16, 32),
                                     "int8", "dhwcn_4", "success"))
# h*w / CORE_NUM > 0
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 12, 35, 17), (3, 2, 16, 16),
                                     "float16", "dhwcn_4", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 12, 2, 4000), (3, 2, 16, 16),
                                     "float16", "dhwcn_5", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 12, 3000, 1), (3, 2, 16, 16),
                                     "float16", "dhwcn_6", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 81, 661, 35, 1), (3, 2, 16, 16),
                                     "float16", "dhwcn_7", "success"))

# c // 16 / CORE_NUM > 0
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 2, 530, 2), (3, 2, 16, 16),
                                     "float16", "dhwcn_8", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 2, 530, 4000), (3, 2, 16, 16),
                                     "float16", "dhwcn_9", "success"))

# d / CORE_NUM > 0
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 3, 2), (3, 2, 16, 16),
                                     "float16", "dhwcn_10", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 35, 2), (3, 2, 16, 16),
                                     "float16", "dhwcn_11", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 2, 4000), (3, 2, 16, 16),
                                     "float16", "dhwcn_12", "success"))

# exception
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 3, 2), (3, 2, 16, 16),
                                     "float32", "dhwcn_13", RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 3, 2), (3, 2, 16, 16),
                                     "float32", "dhwcn_14", RuntimeError,
                                     "FRACTAL_Z"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((35, 3, 2, 3), (3, 2, 16, 16),
                                     "float32", "dhwcn_15", RuntimeError))

# fp32
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 4, 32, 113), (48, 8, 15, 16),
                                     "float32", "dhwcn_fp32_1", RuntimeError))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 4, 32, 113), (48, 8, 16, 17),
                                     "float32", "dhwcn_fp32_2", RuntimeError))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 4, 32, 113), (48, 7, 16, 16),
                                     "float32", "dhwcn_fp32_3", RuntimeError))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 4, 32, 113), (47, 8, 16, 16),
                                     "float32", "dhwcn_fp32_4", RuntimeError))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 32, 16), (2, 1, 16, 16),
                                     "float32", "dhwcn_fp32_5", "success"))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 3, 256, 256), (288, 16, 16, 16),
                                     "float32", "dhwcn_fp32_6", "success"))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 3, 4, 32, 113), (48, 8, 16, 16),
                                     "float32", "dhwcn_fp32_7", "success"))

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1, 1, 1, 32, 17), (2, 2, 16, 16),
                                     "float32", "dhwcn_fp32_8", "success"))

# add precision cases
# c < 16 and n == 1
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 12, 3, 1), (3, 2, 16, 16),
                                                         "float16", "dhwcn_1", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 12, 16, 1), (3, 2, 16, 16),
                                                         "float16", "dhwcn_2", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((3600, 96, 12, 1, 1), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_3", "success"))

# h*w / CORE_NUM > 0
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 12, 35, 17), (3, 2, 16, 16),
                                                         "float16", "dhwcn_4", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((2, 3, 12, 2, 4000), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_5", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((2, 3, 12, 3000, 1), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_6", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((2, 81, 661, 35, 1), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_7", "success"))

# c // 16 / CORE_NUM > 0
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 2, 530, 2), (3, 2, 16, 16),
                                                         "float16", "dhwcn_8", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((2, 3, 2, 530, 4000), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_9", "success"))

# d / CORE_NUM > 0
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((35, 3, 2, 3, 2), (3, 2, 16, 16),
                                                         "float16", "dhwcn_10", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((35, 3, 2, 35, 2), (3, 2, 16, 16),
                                                         "float16", "dhwcn_11", "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((35, 3, 2, 2, 4000), (3, 2, 16, 16),
#                                                         "float16", "dhwcn_12", "success"))

# fp32
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1, 1, 1, 32, 16), (2, 1, 16, 16),
                                                         "float32", "dhwcn_fp32_5", "success"))

ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 3, 256, 256), (288, 16, 16, 16),
                                                         "float32", "dhwcn_fp32_6", "success"))

ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((2, 3, 4, 32, 113), (48, 8, 16, 16),
                                                         "float32", "dhwcn_fp32_7", "success"))

ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1, 1, 1, 32, 17), (2, 2, 16, 16),
                                                         "float32", "dhwcn_fp32_8", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910",
                simulator_mode="pv",
                simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator",
                case_name=["dhwcn_8__1"])

