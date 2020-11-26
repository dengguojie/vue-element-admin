#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format="FRACTAL_Z"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NC1HWC0", "format": "NC1HWC0"},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       "NC1HWC0", dst_format],
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
    axis_c1 = input_shape[1]
    axis_h = input_shape[2]
    axis_w = input_shape[3]
    axis_c0 = input_shape[4]
    axis_ni = 16
    axis_no = _ceil_div(axis_n, axis_ni)
    n_pad = _pad_len(axis_n, axis_ni)

    tmp_input_tensor = np.pad(input_tensor, ((0, n_pad), (0, 0), (0, 0), (0, 0), (0, 0)),
                              mode="constant", constant_values=(0, 0))
    tmp_input_tensor = tmp_input_tensor.reshape(axis_no, axis_ni, axis_c1,
                                                axis_h, axis_w, axis_c0)
    output_tensor = np.transpose(tmp_input_tensor, axes=(2, 3, 4, 0, 1, 5))
    output_shape = dst.get("shape")
    output_tensor = output_tensor.reshape(output_shape)

    return output_tensor


def gen_trans_data_precision_case(src, dst, dtype, case_name_val, expect,
                                  dst_format="FRACTAL_Z"):
    dst = (src[1], src[2], src[3], _ceil_div(src[0], 16), 16, src[4])

    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "NC1HWC0", "format": "NC1HWC0",
                        "param_type": "input", "value_range": [-10.0, 10.0]},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format, "param_type": "output"},
                       "NC1HWC0", dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


# c1*h*w == 1
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10000, 1, 1, 1, 16), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_1", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 1, 1, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_2", "success"))

# n < 3040
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 3, 100, 1, 16), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_3", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 3, 100, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_4", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((100, 3, 100, 1, 16), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_5", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10, 6, 100, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_6", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10, 10, 32, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_7", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((10, 1, 32, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_8", "success"))

# n >= 3040
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((5001, 3, 20, 1, 16), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_9", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((5001, 3, 20, 1, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_10", "success"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3041, 3, 20, 80, 32), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_11", "success"))

# exception
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 1, 1, 1, 16), (3, 2, 16, 16),
                                     "float32", "nc1hwc0_12", RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 1, 1, 1, 15), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_13", RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 1, 1, 1, 16), (3, 2, 16, 16),
                                     "int8", "nc1hwc0_14", RuntimeError))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1000, 1, 1, 16), (3, 2, 16, 16),
                                     "float16", "nc1hwc0_15", RuntimeError))

# add precision case
# c1*h*w == 1
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((10000, 1, 1, 1, 16), (3, 2, 16, 16),
                                                         "float16", "nc1hwc0_1", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1000, 1, 1, 1, 32), (3, 2, 16, 16),
                                                         "int8", "nc1hwc0_2", "success"))

# n < 3040
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1000, 3, 100, 1, 16), (3, 2, 16, 16),
                                                         "float16", "nc1hwc0_3", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((1000, 3, 100, 1, 32), (3, 2, 16, 16),
                                                         "int8", "nc1hwc0_4", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((100, 3, 100, 1, 16), (3, 2, 16, 16),
                                                         "float16", "nc1hwc0_5", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((10, 6, 100, 1, 32), (3, 2, 16, 16),
                                                         "int8", "nc1hwc0_6", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((10, 10, 32, 1, 32), (3, 2, 16, 16),
                                                         "int8", "nc1hwc0_7", "success"))
ut_case.add_precision_case(["Ascend910"],
                           gen_trans_data_precision_case((10, 1, 32, 1, 32), (3, 2, 16, 16),
                                                         "int8", "nc1hwc0_8", "success"))

# n >= 3040
# ut_case.add_precision_case(["Ascend910"],
#                            gen_trans_data_precision_case((5001, 3, 20, 1, 16), (3, 2, 16, 16),
#                                                          "float16", "nc1hwc0_9", "success"))
# ut_case.add_precision_case(["Ascend910"],
#                            gen_trans_data_precision_case((5001, 3, 20, 1, 32), (3, 2, 16, 16),
#                                                          "int8", "nc1hwc0_10", "success"))
# ut_case.add_precision_case(["Ascend910"],
#                            gen_trans_data_precision_case((3041, 3, 20, 80, 32), (3, 2, 16, 16),
#                                                          "int8", "nc1hwc0_11", "success"))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
