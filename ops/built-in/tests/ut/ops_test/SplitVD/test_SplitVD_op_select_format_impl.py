#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SplitVD", "impl.split_v_d", "op_select_format")

def get_split_input(input_shape, split_size, split_dim, split_num, dtype, data_format):
    input_shape = list(input_shape)
    input_dict = {"shape": input_shape, "dtype": dtype, "format": data_format, "ori_shape": input_shape,"ori_format": data_format}
    output_list = []
    for _out_idx in range(split_num):
        output_dim = split_size[_out_idx]
        output_shape = input_shape.copy()
        output_shape[split_dim] = output_dim
        output_dict = {"shape": output_shape, "dtype": dtype, "format": data_format, "ori_shape": output_shape,"ori_format": data_format}
        output_list.append(output_dict)

    params_list = [input_dict, output_list, split_size, split_dim, split_num]

    return params_list


case1 = {"params": get_split_input((128, 128, 128, 128), [64, 64], 0, 2, "float16", "NHWC"),
         "case_name": "split_d_v_op_select_format_5hd_nz",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": get_split_input((128, 128, 128, 127), [64, 63], -1, 2, "float16", "NHWC"),
         "case_name": "split_d_v_op_select_format_nd_c_not_16",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params": get_split_input((128, 128, 127), [64, 63], -1, 2, "float16", "NHWC"),
         "case_name": "split_d_v_op_select_format_nd",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case4 = {"params": get_split_input((128, 128, 128, 8), [4, 4], -1, 2, "float16", "NHWC"),
         "case_name": "split_d_v_op_select_format_other_5hd",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
