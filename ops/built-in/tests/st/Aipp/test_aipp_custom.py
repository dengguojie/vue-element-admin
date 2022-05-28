import json
from threading import local

aipp_config_dict = {
    "aipp_mode": "static",
    "related_input_rank": 0,
    "input_format": "YUV420SP_U8",
    "src_image_size_n": 1,
    "src_image_size_c": 3,
    "src_image_size_h": 418,
    "src_image_size_w": 416,
    "cpadding_value": 0,
    "crop": 0,
    "load_start_pos_h": 16,
    "load_start_pos_w": 16,
    "crop_size_h": 224,
    "crop_size_w": 224,
    "resize": 0,
    "resize_model": 0,
    "resize_output_h": 415,
    "resize_output_w": 415,
    "padding": 0,
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
    "single_line_mode": 0
}
aipp_config = json.dumps(aipp_config_dict)


def gen_static_aipp_case(input_shape, output_shape, dtype_x, dtype_y, format, output_format, aipp_config_json,
                         case_name_val, expect):
    return {
        "params": [{
            "shape": input_shape,
            "dtype": dtype_x,
            "ori_shape": input_shape,
            "ori_format": format,
            "format": format
        }, None, {
            "shape": output_shape,
            "dtype": dtype_y,
            "ori_shape": output_shape,
            "ori_format": format,
            "format": output_format
        }, aipp_config_json],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


def test_aipp_get_op_support_info():
    from impl.aipp import get_op_support_info
    get_op_support_info(
        {
            "shape": (1, 3, 418, 416),
            "dtype": "uint8",
            "format": "NCHW",
            "ori_shape": (1, 3, 418, 416),
            "ori_format": "NCHW"
        }, None, {
            "shape": (1, 1, 418, 416, 32),
            "dtype": "uint8",
            "format": "NC1HWC0",
            "ori_shape": (1, 3, 418, 416),
            "ori_format": "NCHW"
        }, aipp_config)


def test_aipp():
    from tbe.common.context import op_context
    from te import tvm
    from te.platform import cce_conf
    from impl.aipp import aipp

    def aipp_template():
        param_dict = gen_static_aipp_case((1, 3, 418, 416), (1, 1, 418, 416, 32), "uint8", "uint8", "NCHW", "NC1HWC0",
                                          aipp_config, "aipp", "success")
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]

        try:
            aipp(data, input_dync_param, output_data, aipp_config, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["crop"] = 1
        aipp_config_dict["padding"] = 1
        aipp_config1 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 3, 418, 416), (1, 1, 258, 240, 32), "uint8", "uint8", "NCHW", "NC1HWC0",
                                          aipp_config1, "aipp_1", "success")
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config1, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["crop"] = 1
        aipp_config_dict["padding"] = 1
        aipp_config_dict["padding_value"] = 10
        aipp_config2 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 3, 418, 416), (1, 1, 258, 240, 32), "uint8", "uint8", "NCHW", "NC1HWC0",
                                          aipp_config2, "aipp_2", "success")
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config2, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "RGB888_U8"
        aipp_config3 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 3, 418, 416), (1, 1, 258, 240, 32), "uint8", "uint8", "NCHW", "NC1HWC0",
                                          aipp_config3, "aipp_3", "success")
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config3, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "YUV400_U8"
        aipp_config_dict["csc_switch"] = 0
        aipp_config4 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config4, "aipp_4", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config4, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
        aipp_config_dict["csc_switch"] = 0
        aipp_config_dict["padding"] = 0
        aipp_config_dict["resize"] = 1
        aipp_config5 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config5, "aipp_5", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config5, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
        aipp_config_dict["csc_switch"] = 0
        aipp_config_dict["padding"] = 2
        aipp_config_dict["resize"] = 2
        aipp_config6 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config6, "aipp_6", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config6, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
        aipp_config_dict["csc_switch"] = 0
        aipp_config_dict["padding"] = 2
        aipp_config_dict["resize"] = 2
        aipp_config7 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config7, "aipp_7", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config7, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
        aipp_config_dict["csc_switch"] = 0
        aipp_config_dict["padding"] = 2
        aipp_config_dict["resize"] = 2
        aipp_config8 = json.dumps(aipp_config_dict)
        param_dict = gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NC1HWC0_C04",
                                          "NC1HWC0", aipp_config8, "aipp_8", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config8, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict9 = aipp_config_dict.copy()
        del aipp_config_dict9["input_format"]
        aipp_config9 = json.dumps(aipp_config_dict9)
        param_dict = gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NC1HWC0_C04",
                                          "NC1HWC0", aipp_config9, "aipp_9", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config9, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict10 = aipp_config_dict.copy()
        aipp_config_dict10["input_format"] = "RGB888_U8"
        aipp_config10 = json.dumps(aipp_config_dict10)
        param_dict = gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NC1HWC0_C04",
                                          "NC1HWC0", aipp_config10, "aipp_10", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config10, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11 = aipp_config_dict.copy()
        aipp_config_dict11["input_format"] = "YUV400_U8"
        aipp_config_dict11["csc_switch"] = 1
        aipp_config11 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config11, "aipp_11", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config11, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11["input_format"] = "RGB24"
        aipp_config12 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 4, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config12, "aipp_12", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config12, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11["input_format"] = "RGB24_IR"
        aipp_config13 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config13, "aipp_13", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config13, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11["input_format"] = "RAW16"
        aipp_config14 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config14, "aipp_14", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config14, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11["input_format"] = "NC1HWC0DI_FP16"
        aipp_config15 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                          aipp_config15, "aipp_15", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config15, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11 = aipp_config_dict.copy()
        aipp_config_dict11["input_format"] = "NC1HWC0DI_S8"
        aipp_config_dict11["csc_switch"] = 0
        aipp_config16 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "uint8", "float16", "NCHW",
                                          "NC1HWC0", aipp_config16, "aipp_16", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config16, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        param_dict = gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "float16", "NCHW",
                                          "NC1HWC0", aipp_config16, "aipp_17", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config16, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        aipp_config_dict11["input_format"] = "NC1HWC0DI_FP16"
        aipp_config18 = json.dumps(aipp_config_dict11)
        param_dict = gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "uint8", "float16", "NCHW",
                                          "NC1HWC0", aipp_config18, "aipp_18", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config18, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

        param_dict = gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "float16", "int8", "NCHW",
                                          "NC1HWC0", aipp_config18, "aipp_19", RuntimeError)
        data = param_dict["params"][0]
        input_dync_param = param_dict["params"][1]
        output_data = param_dict["params"][2]
        try:
            aipp(data, input_dync_param, output_data, aipp_config18, kernel_name="aipp")
        except Exception as e:
            print("1981 aipp test mock")

    def test_check_aipp_dtype_001():
        aipp_config_dict["input_format"] = "RGB16"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (1, 1, 418, 416, 4), "uint16", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RGB20"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (1, 1, 418, 416, 4), "uint32", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RGB8_IR"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RAW8"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RAW10"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["resize"] = 1
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 418, 416), (1, 1, 258, 240, 32), "uint8", "uint8", "NCHW", "NC1HWC0",
                                         aipp_config, "aipp_24", "success"))["params"]))
        except RuntimeError as e:
            pass
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 3), (
                1, 1, 258, 25, 32), "uint8", "uint8", "NHWC", "NC1HWC0", aipp_config, "aipp_24", "success"))["params"]))
        except RuntimeError as e:
            pass
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 3), (1, 1, 258, 25, 4), "uint8", "uint8", "NHWC", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", "success"))["params"]))
        except RuntimeError as e:
            pass
        aipp_config_dict["input_format"] = "RGB16"
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 3), (1, 3, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 3), (1, 3, 258, 25, 4), "uint8", "uint8", "NHWC", "NC1HWC0_C04",
                                         aipp_config, "aipp_23", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["csc_switch"] = 1
        del aipp_config_dict["resize"]
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "YUV420SP_U8"
        aipp_config_dict["src_image_size_w"] = 415
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "YUV422SP_U8"
        aipp_config_dict["src_image_size_w"] = 415
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 1, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "uint16"
        aipp_config_dict["src_image_size_w"] = 415
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 1, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RGB888_U8"
        aipp_config_dict["src_image_size_w"] = 415
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "XRGB8888_U8"
        aipp_config_dict["src_image_size_w"] = 415
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 4, 258, 4), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "YUYV_U8"
        aipp_config_dict["src_image_size_w"] = 2
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

        aipp_config_dict["input_format"] = "RGB8_IR"
        aipp_config_dict["src_image_size_w"] = 2
        aipp_config = json.dumps(aipp_config_dict)
        try:
            aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                         aipp_config, "aipp_24", RuntimeError))["params"]))
        except RuntimeError as e:
            pass

    def test_aipp_dynamic():

        def gen_dynamic_aipp_case(input_shape, output_shape, dtype_x, dtype_y, format, output_format, aipp_config_json,
                                  case_name_val, expect):
            return {
                "params": [{
                    "shape": input_shape,
                    "dtype": dtype_x,
                    "ori_shape": input_shape,
                    "ori_format": format,
                    "format": format
                }, {
                    "shape": (10000,),
                    "dtype": "uint8",
                    "ori_shape": (10000,),
                    "ori_format": "ND",
                    "format": "ND"
                }, {
                    "shape": output_shape,
                    "dtype": dtype_y,
                    "ori_shape": output_shape,
                    "ori_format": format,
                    "format": output_format
                }, aipp_config_json],
                "case_name": case_name_val,
                "expect": expect,
                "format_expect": [],
                "support_expect": True
            }

        aipp_config_dict_dynamic = {
            "aipp_mode": "dynamic",
            "related_input_rank": 0,
            "input_format": "YUV400_U8",
            "max_src_image_size": 921600
        }
        aipp_config_dynamic = json.dumps(aipp_config_dict_dynamic)
        try:
            aipp(*((gen_dynamic_aipp_case((1, 1, 224, 224), (1, 1, 223, 223, 16), "uint8", "float16", "NCHW", "NC1HWC0",
                                          aipp_config_dynamic, "aipp_dynamic_1", "success"))["params"]))
        except RuntimeError as e:
            pass

    with op_context.OpContext():
        TEST_PLATFORM = ["Ascend320", "Ascend910", "Ascend710", "SD3403"]
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            aipp_template()
            test_check_aipp_dtype_001()
            test_aipp_dynamic()


def test_get_spr9_001():
    from impl.aipp_comm import get_spr9
    aipp_config_dict17 = aipp_config_dict.copy()
    aipp_config_dict17["input_format"] = "NC1HWC0DI_FP16"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "ARGB8888_U8"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "XRGB8888_U8"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "YUYV_U8"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "YUV422SP_U8"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RAW12"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RAW16"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB16"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB20"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB24"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB8_IR"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB16_IR"
    get_spr9(aipp_config_dict17, "float16")
    aipp_config_dict17["input_format"] = "RGB24_IR"
    aipp_config_dict17["raw_rgbir_to_f16_n"] = 0
    get_spr9(aipp_config_dict17, "float16", "NC1HWC0_C04")


def test_get_spr2_spr9_001():
    from impl.aipp_comm import get_spr2_spr9
    from impl.aipp_comm import Const
    if Const.DEFAULT_MATRIX_R0C1_YUV2RGB == 516:
        print("Const.DEFAULT_MATRIX_R0C1_YUV2RGB == 516")
    if Const.DEFAULT_MATRIX_R0C1_YUV2RGB != 516:
        print("Const.DEFAULT_MATRIX_R0C1_YUV2RGB != 516")

    aipp_config_dict18 = aipp_config_dict.copy()
    aipp_map = {}
    aipp_config_dict18["input_format"] = "YUV400_U8"
    aipp_config_dict18["csc_switch"] = 0
    get_spr2_spr9(aipp_config_dict18, "float16", "SD3403", "NCHW", aipp_map)
    aipp_config_dict18["input_format"] = "RAW12"
    get_spr2_spr9(aipp_config_dict18, "float16", "SD3403", "NCHW", aipp_map)
    aipp_config_dict18["csc_switch"] = 1
    aipp_config_dict18["input_format"] = "YUV400_U8"
    get_spr2_spr9(aipp_config_dict18, "float16", "SD3403", "NCHW", aipp_map)
    get_spr2_spr9(aipp_config_dict18, "float16", "SD3400", "NCHW", aipp_map)


def test_set_aipp_default_params_001():
    from impl.aipp_comm import set_aipp_default_params
    aipp_config_dict19 = aipp_config_dict.copy()
    aipp_config_dict19["csc_switch"] = 1
    del aipp_config_dict19["matrix_r0c0"]
    del aipp_config_dict19["matrix_r0c1"]
    del aipp_config_dict19["matrix_r0c2"]
    del aipp_config_dict19["matrix_r1c0"]
    del aipp_config_dict19["matrix_r1c1"]
    del aipp_config_dict19["matrix_r1c2"]
    del aipp_config_dict19["matrix_r2c0"]
    del aipp_config_dict19["matrix_r2c1"]
    del aipp_config_dict19["matrix_r2c2"]
    del aipp_config_dict19["input_bias_0"]
    del aipp_config_dict19["input_bias_1"]
    del aipp_config_dict19["input_bias_2"]
    set_aipp_default_params(aipp_config_dict19)


if __name__ == '__main__':
    test_aipp_get_op_support_info()
    test_aipp()
    test_get_spr9_001()
    test_get_spr2_spr9_001()
    test_set_aipp_default_params_001()
