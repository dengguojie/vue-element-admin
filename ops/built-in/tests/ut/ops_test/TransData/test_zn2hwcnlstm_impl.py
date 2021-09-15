#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, dst_dtype, case_name_val, expect,
                        dst_format="HWCN", src_format="FRACTAL_ZN_LSTM"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                       {"shape": dst, "dtype": dst_dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def gen_trans_data_precision_case(src, dst, dtype, dst_dtype, precision, dst_format="HWCN", src_format="FRACTAL_ZN_LSTM",):
    return {"params": [{"shape": src, "dtype": dtype, "format": "FRACTAL_ZN_LSTM", "ori_shape": src,
                        "ori_format": "FRACTAL_ZN_LSTM", "param_type": "input"},
                       {"shape": dst, "dtype": dst_dtype, "format": dst_format, "ori_shape": dst,
                        "ori_format": dst_format, "param_type": "output"},
                       src_format, dst_format],
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(precision, precision)}

#normal
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((4, 8, 16, 16), (1, 1, 62, 120), "float16", "float16", "case_1", "success"))
#
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((127, 4, 16, 16), (1, 1, 2016, 60), "float16", "float16", "case_2", "success"))
#
#ut_case.add_case(["Ascend910"],
#                gen_trans_data_case((2, 4, 16, 16), (1, 1, 31, 60), "float16", "float16", "case_3", "success"))
#
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 4), "float32", "float32", "case_4", "success"))
#
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((4, 8, 16, 16), (1, 1, 64, 128), "float32", "float32", "case_5", "success"))
#
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((8, 8, 16, 16), (1, 1, 128, 128), "float32", "float32", "case_6", "success"))
#
## exception
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 4),
#                                     "float16", "float32","err_1",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 4),
#                                     "int8", "int8","err_2",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 4),
#                                     "float16", "float16", "err_3",
#                                     RuntimeError,
#                                     src_format="NHWC"))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 4),
#                                     "float16", "float16", "err_4",
#                                     RuntimeError,
#                                     dst_format="NC1HWC0"))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 1, 16), (1, 1, 18, 4),
#                                     "float16", "float16", "err_5",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 4, 18, 4),
#                                     "float16", "float16", "err_6",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16, 5), (1, 1, 18, 4),
#                                     "float16", "float16", "err_7",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 18, 5),
#                                     "float16", "float16", "err_8",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((3, 4, 16, 16), (1, 1, 4, 4),
#                                     "float16", "float16", "err_9",
#                                     RuntimeError))
#ut_case.add_case(["Ascend910"],
#                 gen_trans_data_case((2, 4, 16, 16), (1, 1, 18, 4),
#                                     "float16", "float16", "err_10",
#                                     RuntimeError))


def calc_expect_func(src, dst, src_format, dst_format):
    src_shape = src["shape"]
    src_dtype = src["dtype"]
    out_shape = dst["shape"]
    out_dtype = dst["dtype"]

    c0 =16
    h_in = out_shape[3] // 4
    h_align = math.ceil(h_in / c0) * c0
    i = out_shape[2] - h_in
    i_align = math.ceil(i / c0) * c0

    src_data = src["value"]
    out_data = np.zeros((out_shape[0], out_shape[1], out_shape[2], out_shape[3]), dtype=out_dtype)

    for h in range(out_shape[0]):
        for w in range(out_shape[1]):
            for c in range(out_shape[2]):
                if c < i:
                    for m in range(4):
                        for n in range(h_in):
                            num_c1 = c // c0
                            num_c0 = c % c0
                            num_n0 = (m * h_align + n) // c0
                            num_ni = n % c0
                            out_data[h][w][c][m*h_in+n] = src_data[num_c1*out_shape[0]*out_shape[1]+h*out_shape[1]+w][num_n0][num_ni][num_c0]
                else:
                    for m in range(4):
                        for n in range(h_in):
                            num_c1 = (c - i + i_align) // c0
                            num_c0 = (c - i + i_align) % c0
                            num_n0 = (m * h_align + n) // c0
                            num_ni = n % c0
                            out_data[h][w][c][m*h_in+n] = src_data[num_c1*out_shape[0]*out_shape[1]+h*out_shape[1]+w][num_n0][num_ni][num_c0]

    return out_data


# ut_case.add_precision_case("all", gen_trans_data_precision_case((4, 8, 16, 16), (1, 1, 62, 120), "float16", "float16", 0.001))
#
# ut_case.add_precision_case("all", gen_trans_data_precision_case((127, 4, 16, 16), (1, 1, 2016, 60), "float16", "float16", 0.001))
#
# ut_case.add_precision_case("all", gen_trans_data_precision_case((2, 4, 16, 16), (1, 1, 31, 60), "float16", "float16", 0.001))
#
# ut_case.add_precision_case("all", gen_trans_data_precision_case((3, 4, 16, 16), (1, 1, 18, 4), "float32", "float32", 0.0001))
#
# ut_case.add_precision_case("all", gen_trans_data_precision_case((4, 8, 16, 16), (1, 1, 64, 128), "float32", "float32", 0.0001))
#
# ut_case.add_precision_case("all", gen_trans_data_precision_case((8, 8, 16, 16), (1, 1, 128, 128), "float32", "float32", 0.0001))


if __name__ == '__main__':
    ut_case.run()
    exit(0)

