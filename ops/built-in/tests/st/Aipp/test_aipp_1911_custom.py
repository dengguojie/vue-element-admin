#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import json
import te.platform as tbe_platform


aipp_config_dict = {
    "aipp_mode": "static",
    "related_input_rank": 0,
    "input_format": "YUV420SP_U8",
    "src_image_size_n": 1,
    "src_image_size_c": 3,
    "src_image_size_h": 418,
    "src_image_size_w": 416,
    "cpadding_value": 0,
    "crop": 1,
    "load_start_pos_h": 16,
    "load_start_pos_w": 16,
    "crop_size_h": 224,
    "crop_size_w": 224,
    "padding": 1,
    "left_padding_size": 2,
    "right_padding_size": 14,
    "top_padding_size": 2,
    "bottom_padding_size": 32,
    "csc_switch": 1,
    "rbuv_swap_switch": 0,
    "matrix_r0c0": 256,
    "matrix_r0c1": 454,
    "matrix_r0c2": 0,
    "matrix_r1c0": 256,
    "matrix_r1c2": -183,
    "matrix_r1c1": -88,
    "matrix_r2c0": 256,
    "matrix_r2c1": 0,
    "matrix_r2c2": 359,
    "input_bias_0": 0,
    "input_bias_1": 128,
    "input_bias_2": 128,
    "min_chn_0": 0,
    "min_chn_1": 0,
    "min_chn_2": 0,
    "min_chn_3": 0,
    "mean_chn_0": 0,
    "mean_chn_1": 0,
    "mean_chn_2": 0,
    "mean_chn_3": 0,
    "ax_swap_switch": 0,
    "single_line_mode": 0,
}


def gen_static_aipp_case(
    input_shape, output_shape, dtype_x, dtype_y, format, output_format, aipp_config_json, case_name_val, expect
):
    return {
        "params": [
            {"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": format, "format": format},
            None,
            {
                "shape": output_shape,
                "dtype": dtype_y,
                "ori_shape": output_shape,
                "ori_format": format,
                "format": output_format,
            },
            aipp_config_json,
        ],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def test_aipp_static_1():
    param_dict = gen_static_aipp_case(
        (1, 3, 418, 416),
        (1, 1, 258, 240, 32),
        "uint8",
        "uint8",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict,
        "aipp_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


def test_aipp_static_2():
    aipp_config_dict_tmp = aipp_config_dict.copy()
    aipp_config_dict_tmp["input_format"] = "RGB888_U8"
    param_dict = gen_static_aipp_case(
        (1, 3, 418, 416),
        (1, 1, 258, 240, 32),
        "uint8",
        "uint8",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict_tmp,
        "aipp_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


def test_aipp_static_3():
    aipp_config_dict_tmp = aipp_config_dict.copy()
    aipp_config_dict_tmp["input_format"] = "YUV422SP_U8"
    param_dict = gen_static_aipp_case(
        (1, 3, 418, 416),
        (1, 1, 258, 240, 32),
        "uint8",
        "uint8",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict_tmp,
        "aipp_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


def test_aipp_static_4():
    aipp_config_dict_tmp = aipp_config_dict.copy()
    aipp_config_dict_tmp["crop_size_h"] = 10
    aipp_config_dict_tmp["crop_size_w"] = 10
    param_dict = gen_static_aipp_case(
        (1, 3, 418, 416),
        (1, 1, 44, 26, 32),
        "uint8",
        "uint8",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict_tmp,
        "aipp_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


def test_aipp_static_5():
    aipp_config_dict_tmp = aipp_config_dict.copy()
    aipp_config_dict_tmp["crop_size_h"] = 10
    aipp_config_dict_tmp["crop_size_w"] = 10
    param_dict = gen_static_aipp_case(
        (1, 3, 418, 416),
        (1, 1, 44, 26, 32),
        "uint8",
        "float16",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict_tmp,
        "aipp_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_tmp, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


def gen_dynamic_aipp_case(
    input_shape, output_shape, dtype_x, dtype_y, format, output_format, aipp_config_json, case_name_val, expect
):
    return {
        "params": [
            {"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": format, "format": format},
            {"shape": (10000,), "dtype": "uint8", "ori_shape": (10000,), "ori_format": "ND", "format": "ND"},
            {
                "shape": output_shape,
                "dtype": dtype_y,
                "ori_shape": output_shape,
                "ori_format": format,
                "format": output_format,
            },
            aipp_config_json,
        ],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


aipp_config_dict_dynamic = {
    "aipp_mode": "dynamic",
    "related_input_rank": 0,
    "input_format": "YUV400_U8",
    "max_src_image_size": 921600,
}


def test_aipp_dynamic_1():
    aipp_config_dict_dynamic_tmp = aipp_config_dict_dynamic.copy()
    aipp_config_dict_dynamic_tmp["input_format"] = "YUV420SP_U8"
    param_dict = gen_dynamic_aipp_case(
        (1, 1, 224, 224),
        (1, 1, 224, 224, 16),
        "uint8",
        "float16",
        "NCHW",
        "NC1HWC0",
        aipp_config_dict_dynamic_tmp,
        "aipp_dynamic_1",
        "success",
    )
    input_data = param_dict["params"][0]
    input_dync_param = param_dict["params"][1]
    output_data = param_dict["params"][2]
    from impl.aipp_stc_dyn import new_aipp_compute

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_dynamic_tmp, "Ascend920", kernel_name="aipp")
    except Exception as e:
        print("1981 aipp test mock")

    try:
        new_aipp_compute(input_data, input_dync_param, output_data, aipp_config_dict_dynamic_tmp, "Ascend320", kernel_name="aipp")
    except Exception as e:
        print("1911 aipp test mock")


if __name__ == "__main__":
    test_aipp_static_1()
    test_aipp_static_2()
    test_aipp_static_3()
    test_aipp_static_4()
    test_aipp_static_5()
    test_aipp_dynamic_1()
    print("end of aipp test")