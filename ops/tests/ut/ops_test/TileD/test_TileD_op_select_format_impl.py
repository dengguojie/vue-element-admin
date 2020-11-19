#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("TileD", "impl.tile_d", "op_select_format")

def get_input(input_shape, multiples, dtype, data_format):
    input_shape = list(input_shape)
    input_dict = {"shape": input_shape, "dtype": dtype, "format": data_format, "ori_shape": input_shape,"ori_format": data_format}
    output_shape = []
    for _out_idx in range(len(input_shape)):
        output_dim = input_shape[_out_idx] * multiples[_out_idx]
        output_shape.append(output_dim)
    output_dict = {"shape": output_shape, "dtype": dtype, "format": data_format, "ori_shape": output_shape,"ori_format": data_format}

    params_list = [input_dict, output_dict, multiples]

    return params_list


case1 = {"params": get_input((128, 1, 1, 1), [16, 16, 16, 16], "float16", "NHWC"),
         "case_name": "tile_d_op_select_format_4d_2_5hd",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

case2 = {"params": get_input((128, 128, 128, 128), [16, 16, 16, 16], "float16", "NHWC"),
         "case_name": "tile_d_op_select_format_all_format",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
