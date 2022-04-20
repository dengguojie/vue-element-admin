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


def calc_expect_func(src, dst, src_format, dst_format):
    input_shape = src.get("shape")
    input_tensor = src.get("value")
    axis_h = input_shape[0]
    axis_w = input_shape[1]
    axis_n = input_shape[3] // 4
    axis_c_p1 = input_shape[2] - axis_n
    axis_c_p2 = axis_n
    axis_ni = 16
    axis_c0 = 16
    axis_c1_p1 = _ceil_div(axis_c_p1, axis_c0)
    axis_c1_p2 = _ceil_div(axis_c_p2, axis_c0)
    axis_no = _ceil_div(axis_n, axis_ni)
    c_pad_1 = _pad_len(axis_c_p1, axis_c0)
    c_pad_2 = _pad_len(axis_c_p2, axis_c0)
    n_pad = _pad_len(axis_n, axis_ni)

    input_tensor_n_part1 = input_tensor[:, :, :, :axis_n]
    input_tensor_n_part2 = input_tensor[:, :, :, axis_n:axis_n * 2]
    input_tensor_n_part3 = input_tensor[:, :, :, axis_n * 2:axis_n * 3]
    input_tensor_n_part4 = input_tensor[:, :, :, axis_n * 3:axis_n * 4]

    tmp_input_tensor_n_part1 = np.pad(input_tensor_n_part1, ((0, 0), (0, 0), (0, 0), (0, n_pad)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor_n_part2 = np.pad(input_tensor_n_part2, ((0, 0), (0, 0), (0, 0), (0, n_pad)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor_n_part3 = np.pad(input_tensor_n_part3, ((0, 0), (0, 0), (0, 0), (0, n_pad)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor_n_part4 = np.pad(input_tensor_n_part4, ((0, 0), (0, 0), (0, 0), (0, n_pad)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor = np.concatenate((tmp_input_tensor_n_part1, tmp_input_tensor_n_part2,
                                       tmp_input_tensor_n_part3, tmp_input_tensor_n_part4), axis=3)
    input_tensor_c_part1 = tmp_input_tensor[:, :, :axis_c_p1, :]
    input_tensor_c_part2 = tmp_input_tensor[:, :, axis_c_p1:axis_c_p1 + axis_n, :]
    tmp_input_tensor_c_part1 = np.pad(input_tensor_c_part1, ((0, 0), (0, 0), (0, c_pad_1), (0, 0)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor_c_part2 = np.pad(input_tensor_c_part2, ((0, 0), (0, 0), (0, c_pad_2), (0, 0)),
                                      mode="constant", constant_values=(0, 0))
    tmp_input_tensor = np.concatenate((tmp_input_tensor_c_part1, tmp_input_tensor_c_part2), axis=2)
    tmp_input_tensor = tmp_input_tensor.reshape(axis_h, axis_w,
                                                axis_c1_p1 + axis_c1_p2, axis_c0,
                                                4 * axis_no, axis_ni)
    output_tensor = np.transpose(tmp_input_tensor, axes=(2, 0, 1, 4, 5, 3))
    output_shape = dst.get("shape")
    output_tensor = output_tensor.reshape(output_shape)

    return output_tensor


def gen_trans_data_precision_case(src, dst, dtype, case_name_val, expect,
                                  dst_format="FRACTAL_ZN_LSTM"):
    axis_h = src[0]
    axis_w = src[1]
    axis_n = src[3] // 4
    axis_c_p1 = src[2] - axis_n
    axis_c_p2 = axis_n
    axis_ni = 16
    axis_c0 = 16
    axis_no = _ceil_div(axis_n, axis_ni)
    axis_c1_p1 = _ceil_div(axis_c_p1, axis_c0)
    axis_c1_p2 = _ceil_div(axis_c_p2, axis_c0)
    dst = ((axis_c1_p1 + axis_c1_p2)*axis_h*axis_w, 4 * axis_no, axis_ni, axis_c0)

    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "HWCN", "format": "HWCN", "param_type": "input",
                        "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format, "param_type": "output"},
                       "HWCN", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


# add precision cases
ut_case.add_precision_case(["Ascend910A"],
                           gen_trans_data_precision_case((1, 1, 12, 12), (3, 2, 16, 16),
                                                         "float16", "hwcn2fzlstm_1", "success"))
ut_case.add_precision_case(["Ascend910A"],
                           gen_trans_data_precision_case((1, 1, 32, 64), (3, 2, 16, 16),
                                                         "float16", "hwcn2fzlstm_2", "success"))
ut_case.add_precision_case(["Ascend910A"],
                           gen_trans_data_precision_case((1, 1, 510, 1240), (3, 2, 16, 16),
                                                         "float16", "hwcn2fzlstm_4", "success"))
ut_case.add_precision_case(["Ascend910A"],
                           gen_trans_data_precision_case((1, 1, 201, 800), (3, 2, 16, 16),
                                                         "float16", "hwcn2fzlstm_5", "success"))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
