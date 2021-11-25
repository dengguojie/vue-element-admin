#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("TileD", "impl.tile_d", "tile_d")


def trans_shape(shape, ori_format, need_format):
    new_shape = shape
    if need_format == "FRACTAL_Z":
        dict_shape = dict(zip(list(ori_format), shape))
        shape_c1 = dict_shape["C"] + 15 // 16
        shape_n1 = dict_shape["N"] + 15 // 16
        shape_h = dict_shape["H"]
        shape_w = dict_shape["W"]
        new_shape = [shape_c1*shape_h*shape_w, shape_c1, 16, 16]

    return new_shape


def get_input(input_shape, multiples, dtype, ori_format, input_format):
    input_shape = list(input_shape)
    new_shape = trans_shape(input_shape, ori_format, input_format)
    input_dict = {"shape": new_shape, "dtype": dtype, "format": input_format, "ori_shape": input_shape,"ori_format": ori_format}
    output_shape = []
    for _out_idx in range(len(input_shape)):
        output_dim = input_shape[_out_idx] * multiples[_out_idx]
        output_shape.append(output_dim)
    output_shape = trans_shape(output_shape, ori_format, input_format)
    output_dict = {"shape": output_shape, "dtype": dtype, "format": input_format, "ori_shape": output_shape,"ori_format": ori_format}

    params_list = [input_dict, output_dict, multiples]

    return params_list


case1 = {"params": get_input((128, 128, 128, 128), [16, 16, 16, 16], "float16", "NHWC", "FRACTAL_Z"),
         "case_name": "tile_d_op_select_format_4d_2_5hd",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}


def test_tile_get_op_support_info(test_arg):
    from impl.tile_d import get_op_support_info
    get_op_support_info(
        {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, {
            "shape": (8, 16, 5, 5, 16),
            "dtype": "float16",
            "format": "NC1HWC0",
            "ori_shape": (8, 5, 5, 256),
            "ori_format": "NHWC"
        }, [1,1,1,1])


ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_cust_test_func(test_func=test_tile_get_op_support_info)

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
