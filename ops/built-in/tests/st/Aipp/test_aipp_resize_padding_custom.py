#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import json
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf
from impl import aipp
from impl import aipp_resize_padding


aipp_config_dict = {"aipp_mode":"static",
                    "related_input_rank":0,
                    "input_format":"YUV420SP_U8",
                    "src_image_size_n" : 1,
                    "src_image_size_c" : 3,
                    "src_image_size_h" : 418,
                    "src_image_size_w" : 416,
                    "crop" : 1,
                    "load_start_pos_h" : 16,
                    "load_start_pos_w" : 16,
                    "crop_size_h" : 224,
                    "crop_size_w" : 224,
                    "resize" : 0,
                    "resize_model" : 0,
                    "resize_output_h" : 415,
                    "resize_output_w" : 415,
                    "padding" : 1,
                    "left_padding_size" : 2,
                    "right_padding_size" : 14,
                    "top_padding_size" : 0,
                    "bottom_padding_size" : 32,
                    "csc_switch" : 1,
                    "rbuv_swap_switch":0,
                    "matrix_r0c0":256,
                    "matrix_r0c1":454,
                    "matrix_r0c2":0,
                    "matrix_r1c0":256,
                    "matrix_r1c2":-183,
                    "matrix_r1c1":-88,
                    "matrix_r2c0":256,
                    "matrix_r2c1":0,
                    "matrix_r2c2":359,
                    "input_bias_0":0,
                    "input_bias_1":128,
                    "input_bias_2":128,
                    "min_chn_0":0,
                    "min_chn_1":0,
                    "min_chn_2":0,
                    "min_chn_3":0,
                    "mean_chn_0":0,
                    "mean_chn_1":0,
                    "mean_chn_2":0,
                    "mean_chn_3":0,
                    "reci_chn_0":1,
                    "reci_chn_1":1,
                    "reci_chn_2":1,
                    "reci_chn_3":1,
                    "ax_swap_switch":0,
                    "single_line_mode":0
                    }
aipp_config = json.dumps(aipp_config_dict)


def test_aipp_resize_padding():
    input_shape = (1,3,418,416)
    format = "NCHW"
    input_data = {"shape": input_shape, "dtype": "uint8", "format": format, "ori_shape": input_shape,"ori_format": format}
    output_data = {"shape": (1,1,256,240,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,256,240),"ori_format": format}

    data = tvm.placeholder(input_shape, name='input', dtype="uint8")
    with tbe.common.context.op_context.OpContext():
        aipp_resize_padding.aipp_compute(data, input_shape, format, output_data, aipp_config_dict)

def test_aipp_resize_padding2():
    input_shape = (1,3,418,416)
    format = "NCHW"
    input_data = {"shape": input_shape, "dtype": "uint8", "format": format, "ori_shape": input_shape,"ori_format": format}
    output_data = {"shape": (1,1,256,240,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,3,256,240),"ori_format": format}
    aipp_config_dict["input_format"] = "YUV420SP_U8"

    data = tvm.placeholder(input_shape, name='input', dtype="uint8")
    with tbe.common.context.op_context.OpContext():
        aipp_resize_padding.aipp_compute(data, input_shape, format, output_data, aipp_config_dict)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_aipp_resize_padding()
    test_aipp_resize_padding2()
    cce_conf.te_set_version(soc_version)
