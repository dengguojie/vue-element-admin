#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data",
               "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z_3D"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NCHW", "format": "NCHW"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": "NCHW", "format": "NCHW"},
                       "NCDHW", dst_format],
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
    axis_c = input_shape[1]
    axis_d = input_shape[2]
    axis_h = input_shape[3]
    axis_w = input_shape[4]
    axis_ni = 16
    axis_c0 = 16
    axis_c1 = _ceil_div(axis_c, axis_c0)
    axis_no = _ceil_div(axis_n, axis_ni)
    c_pad = _pad_len(axis_c, axis_c0)
    n_pad = _pad_len(axis_n, axis_ni)

    tmp_input_tensor = np.pad(input_tensor, ((0, n_pad), (0, c_pad), (0, 0), (0, 0), (0, 0)),
                              mode="constant", constant_values=(0, 0))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no, axis_ni, axis_c1, axis_c0,
                                                axis_d, axis_h, axis_w)
    output_tensor = np.transpose(tmp_input_tensor, axes=(4, 2, 5, 6, 0, 1, 3))

    return output_tensor


def gen_trans_data_precision_case(src, dst, dtype, case_name_val,
                                  expect, dst_format="FRACTAL_Z_3D"):
    c0_len = 32 if dtype == "int8" else 16
    dst = (src[2], _ceil_div(src[1], c0_len), src[3], src[4], _ceil_div(src[0], 16), 16, c0_len)

    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NCHW", "format": "NCHW",
                        "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": "NCHW", "format": "NCHW", "param_type": "output"},
                       "NCDHW", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


# network shape
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16, 5, 3, 3, 3), (3, 2, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_001", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16, 5, 3, 3, 3), (3, 2, 16, 16),
                                     "float32", "ncdhw_2_fractal_z_3d_002", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_003", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                     "float32", "ncdhw_2_fractal_z_3d_004", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128, 2, 3, 3, 3), (3, 2, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_005", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((116, 61, 3, 3, 1000), (3, 2, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_006", "success"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((64, 3, 5, 7, 7), (245, 4, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_108", "success"))

# exception
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                     "float16", "ncdhw_2_fractal_z_3d_008", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                     "int8", "ncdhw_2_fractal_z_3d_009", RuntimeError))

# add precision case
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((16, 5, 3, 3, 3), (3, 2, 16, 16),
                                                         "float16", "ncdhw_2_fractal_z_3d_001",
                                                         "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((16, 5, 3, 3, 3), (3, 2, 16, 16),
                                                         "float32", "ncdhw_2_fractal_z_3d_002",
                                                         "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                                         "float16", "ncdhw_2_fractal_z_3d_003",
                                                         "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((128, 64, 3, 3, 3), (3, 2, 16, 16),
                                                         "float32", "ncdhw_2_fractal_z_3d_004",
                                                         "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((128, 2, 3, 3, 3), (3, 2, 16, 16),
                                                         "float16", "ncdhw_2_fractal_z_3d_005",
                                                         "success"))
#ut_case.add_precision_case(["Ascend910"],
#                           gen_trans_data_precision_case((116, 61, 3, 3, 1000), (3, 2, 16, 16),
#                                                         "float16", "ncdhw_2_fractal_z_3d_006",
#                                                         "success"))
#ut_case.add_precision_case(["Ascend910"],
#                 gen_trans_data_precision_case((116, 61, 3, 3, 1000), (3, 2, 16, 16),
#                                               "float32", "ncdhw_2_fractal_z_3d_007", "success"))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
