#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import json
ut_case = OpUT("Aipp", "impl.aipp", "aipp")

aipp_config_dict = {"aipp_mode":"static",
                    "related_input_rank":0,
                    "input_format":"YUV420SP_U8",
                    "src_image_size_n" : 1,
                    "src_image_size_c" : 3,
                    "src_image_size_h" : 418,
                    "src_image_size_w" : 416,
                    "cpadding_value" : 0,
                    "crop" : 0,
                    "load_start_pos_h" : 16,
                    "load_start_pos_w" : 16,
                    "crop_size_h" : 224,
                    "crop_size_w" : 224,
                    "resize" : 0,
                    "resize_model" : 0,
                    "resize_output_h" : 415,
                    "resize_output_w" : 415,
                    "padding" : 0,
                    "left_padding_size" : 2,
                    "right_padding_size" : 14,
                    "top_padding_size" : 2,
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
                    "ax_swap_switch":0,
                    "single_line_mode":0
                    }
aipp_config = json.dumps(aipp_config_dict)

def gen_static_aipp_case(input_shape, output_shape, dtype_x, dtype_y, format, output_format,
                         aipp_config_json, case_name_val, expect):
    return {"params": [{"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": format, "format": format},
                       None,
                       {"shape": output_shape, "dtype": dtype_y, "ori_shape": output_shape, "ori_format": format, "format": output_format},
                       aipp_config_json],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


def test_aipp_get_op_support_info(test_arg):
    from impl.aipp import get_op_support_info
    get_op_support_info({"shape": (1,3,418,416), "dtype": "uint8", "format": "NCHW", "ori_shape": (1,3,418,416),"ori_format": "NCHW"}, 
                        None,
                        {"shape": (1,1,418,416,32), "dtype": "uint8", "format": "NC1HWC0", "ori_shape": (1,3,418,416),"ori_format": "NCHW"},
                        aipp_config)


ut_case.add_cust_test_func(test_func=test_aipp_get_op_support_info)

ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,418,416,32),
                                      "uint8", "uint8", "NCHW", "NC1HWC0", aipp_config, "aipp_1", "success"))
ut_case.add_case(["Ascend910"],
                 gen_static_aipp_case((1,3,418,416), (1,1,418,416,32),
                                      "uint8", "uint8", "NCHW", "NC1HWC0", aipp_config, "aipp_2", RuntimeError))

ut_case.add_case(["Ascend310"],
                 gen_static_aipp_case((1,3,418,416), (1,1,418,416,4),
                                      "uint8", "uint8", "NCHW", "NC1HWC0_C04", aipp_config, "aipp_c04_1", RuntimeError))

aipp_config_dict["crop"] = 1
aipp_config_dict["padding"] = 1
aipp_config3 = json.dumps(aipp_config_dict)
aipp_config_dict21 = aipp_config_dict.copy()
ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,258,240,32),
                                      "uint8", "uint8", "NCHW", "NC1HWC0", aipp_config3, "aipp_3", "success"))


aipp_config_dict["crop"] = 1
aipp_config_dict["padding"] = 1
aipp_config_dict["padding_value"] = 10
aipp_config4 = json.dumps(aipp_config_dict)

ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,258,240,32),
                                      "uint8", "uint8", "NCHW", "NC1HWC0", aipp_config4, "aipp_4", "success"))
ut_case.add_case(["Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,258,240,4),
                                      "uint8", "uint8", "NCHW", "NC1HWC0_C04", aipp_config4, "aipp_c04_2", "success"))

aipp_config_dict["input_format"] = "RGB888_U8"
aipp_config5 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,258,240,16),
                                      "uint8", "float16", "NCHW", "NC1HWC0", aipp_config5, "aipp_6", "success"))

aipp_config_dict["input_format"] = "YUV400_U8"
aipp_config_dict["csc_switch"] = 0
aipp_config6 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,1,418,416), (1,1,258,240,16),
                                      "uint8", "float16", "NCHW", "NC1HWC0", aipp_config6, "aipp_5", "success"))

aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
aipp_config_dict["csc_switch"] = 0
aipp_config_dict["padding"] = 0
aipp_config_dict["resize"] = 1
aipp_config7 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "uint8", "float16", "NCHW", "NC1HWC0",
                                      aipp_config7, "aipp_7", RuntimeError))

aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
aipp_config_dict["csc_switch"] = 0
aipp_config_dict["padding"] = 2
aipp_config_dict["resize"] = 2
aipp_config8 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend910A"],
                 gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "uint8", "float16", "NCHW", "NC1HWC0",
                                      aipp_config8, "aipp_8", RuntimeError))

aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
aipp_config_dict["csc_switch"] = 0
aipp_config_dict["padding"] = 2
aipp_config_dict["resize"] = 2
aipp_config9 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend910A"],
                 gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config9, "aipp_9", RuntimeError))

aipp_config_dict["input_format"] = "NC1HWC0DI_S8"
aipp_config_dict["csc_switch"] = 0
aipp_config_dict["padding"] = 2
aipp_config_dict["resize"] = 2
aipp_config10 = json.dumps(aipp_config_dict)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 258, 240, 4), (1, 1, 258, 240, 4), "int8", "int8", "NC1HWC0_C04",
                                      "NC1HWC0", aipp_config10, "aipp_10", RuntimeError))

aipp_config_dict11 = aipp_config_dict.copy()
del aipp_config_dict11["input_format"]
aipp_config11 = json.dumps(aipp_config_dict11)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NC1HWC0_C04", "NC1HWC0",
                                      aipp_config11, "aipp_11", RuntimeError))

aipp_config_dict12 = aipp_config_dict.copy()
aipp_config_dict12["input_format"] = "RGB888_U8"
aipp_config12 = json.dumps(aipp_config_dict12)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NC1HWC0_C04", "NC1HWC0",
                                      aipp_config12, "aipp_12", RuntimeError))             

aipp_config_dict13 = aipp_config_dict.copy()
aipp_config_dict13["input_format"] = "YUV400_U8"
aipp_config_dict13["csc_switch"] = 1
aipp_config13 = json.dumps(aipp_config_dict13)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 4, 418, 416, 4), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config13, "aipp_13", RuntimeError))   

aipp_config_dict13["input_format"] = "RGB24"
aipp_config14 = json.dumps(aipp_config_dict13)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 4, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config14, "aipp_14", RuntimeError))   

aipp_config_dict13["input_format"] = "RGB24_IR"
aipp_config15 = json.dumps(aipp_config_dict13)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config15, "aipp_15", RuntimeError)) 

aipp_config_dict13["input_format"] = "RAW16"
aipp_config16 = json.dumps(aipp_config_dict13)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config16, "aipp_16", RuntimeError)) 

aipp_config_dict13["input_format"] = "NC1HWC0DI_FP16"
aipp_config17 = json.dumps(aipp_config_dict13)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 3, 418, 416, 3), (1, 4, 258, 240, 4), "int8", "int8", "NCHW", "NC1HWC0",
                                      aipp_config17, "aipp_17", RuntimeError)) 

aipp_config_dict20 = aipp_config_dict.copy()
aipp_config_dict20["input_format"] = "NC1HWC0DI_S8"
aipp_config_dict20["csc_switch"] = 0
aipp_config20 = json.dumps(aipp_config_dict20)

ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "uint8", "float16", "NCHW", "NC1HWC0",
                                      aipp_config20, "aipp_18", RuntimeError))

ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "int8", "float16", "NCHW", "NC1HWC0",
                                      aipp_config20, "aipp_19", RuntimeError))

aipp_config_dict20["input_format"] = "NC1HWC0DI_FP16"
aipp_config21 = json.dumps(aipp_config_dict20)
ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "uint8", "float16", "NCHW", "NC1HWC0",
                                      aipp_config21, "aipp_20", RuntimeError))

ut_case.add_case(["Ascend310", "Ascend910A"],
                 gen_static_aipp_case((1, 1, 418, 416, 4), (1, 1, 418, 416, 4), "float16", "int8", "NCHW", "NC1HWC0",
                                      aipp_config21, "aipp_21", RuntimeError))

def gen_dynamic_aipp_case(input_shape, output_shape, dtype_x, dtype_y, format, output_format,
                          aipp_config_json, case_name_val, expect):
    return {"params": [{"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": format, "format": format},
                       {"shape": (10000,), "dtype": "uint8", "ori_shape": (10000,), "ori_format": "ND", "format": "ND"},
                       {"shape": output_shape, "dtype": dtype_y, "ori_shape": output_shape, "ori_format": format, "format": output_format},
                       aipp_config_json],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

aipp_config_dict_dynamic = {"aipp_mode":"dynamic",
                            "related_input_rank":0,
                            "input_format":"YUV400_U8",
                            "max_src_image_size":921600
                           }
aipp_config_dynamic = json.dumps(aipp_config_dict_dynamic)
ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_dynamic_aipp_case((1,1,224,224), (1,1,223,223,16),
                                       "uint8", "float16", "NCHW", "NC1HWC0", aipp_config_dynamic,
                                       "aipp_dynamic_1", "success"))

from tbe import tvm
def test_set_spr2_spr9_1(test_arg):
    from impl.aipp_comm import set_spr2_spr9
    aipp_config_dict11 = aipp_config_dict.copy()
    aipp_config_dict11["input_format"] = "YUV400_U8"
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict11, "int8", "SD3403", "NC1HWC0")
    aipp_config_dict12 = aipp_config_dict.copy()
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict12, "int8", "SD3403", "NC1HWC0")
    aipp_config_dict13 = aipp_config_dict.copy()
    aipp_config_dict13["csc_switch"] = 1
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict13, "int8", "SD3403", "NC1HWC0")
    aipp_config_dict14 = aipp_config_dict.copy()
    aipp_config_dict14["input_format"] = "RAW10"
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict14, "float16", "SD3403", "NC1HWC0")
    aipp_config_dict15 = aipp_config_dict.copy()
    aipp_config_dict15["padding"] = 1
    aipp_config_dict15["input_format"] = "AYUV444_U8"
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict15, "float16", "SD3403", "NC1HWC0")
    aipp_config_dict16 = aipp_config_dict.copy()
    aipp_config_dict16["input_format"] = "YUV420SP_U8"
    aipp_config_dict16["csc_switch"] = 1
    set_spr2_spr9(tvm.ir_builder.create(), aipp_config_dict16, "float16", "SD3403", "NC1HWC0")

def test_aipp_001(test_arg):
    from te.platform.cce_conf import te_set_version
    from impl.aipp import aipp
    te_set_version("SD3403")
    aipp(
        {
            "shape": (1, 1, 224, 224),
            "dtype": "uint8",
            "ori_shape": (1, 1, 224, 224),
            "ori_format": "NCHW",
            "format": "NCHW"
        }, {
            "shape": (10000,),
            "dtype": "uint8",
            "ori_shape": (10000,),
            "ori_format": "ND",
            "format": "ND"
        }, {
            "shape": (1, 1, 223, 223, 16),
            "dtype": "float16",
            "ori_shape": (1, 1, 223, 223, 16),
            "ori_format": "NCHW",
            "format": "NC1HWC0"
        }, aipp_config_dynamic)
    te_set_version("SD3403")

def test_get_spr9_001(test_arg):
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


def test_get_spr2_spr9_001(test_arg):
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



def test_set_aipp_default_params_001(test_arg):
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


def test_check_aipp_dtype_001(test_arg):
    from impl.aipp import aipp
    from tbe.common.platform.platform_info import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend615")
    aipp_config_dict = aipp_config_dict20.copy()
    aipp_config_dict["input_format"] = "RGB16"
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (
            1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (
            1, 1, 418, 416, 4), "uint16", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    aipp_config_dict["input_format"] = "RGB20"
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (
            1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (
            1, 1, 418, 416, 4), "uint32", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    aipp_config_dict["input_format"] = "RGB8_IR"
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416, 4), (
            1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    set_current_compile_soc_info("Ascend320")
    aipp_config_dict["input_format"] = "RAW8"
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 1, 418, 416, 4), (
            1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    aipp_config_dict["input_format"] = "RAW10"
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 1, 418, 416, 4), (
            1, 1, 418, 416, 4), "int8", "int8", "NCHW", "NC1HWC0", aipp_config, "aipp_23", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    set_current_compile_soc_info("SD3403")
    aipp_config_dict21["resize"] = 1
    aipp_config = json.dumps(aipp_config_dict21)
    try:
        aipp(*((gen_static_aipp_case((1, 3, 418, 416), (
            1, 1, 258, 240, 32), "uint8", "uint8", "NCHW", "NC1HWC0", aipp_config, "aipp_24", "success"))["params"]))
    except RuntimeError as e:
        pass
    try:
        aipp(*((gen_static_aipp_case((1, 3, 258, 3), (
            1, 1, 258, 25, 32), "uint8", "uint8", "NHWC", "NC1HWC0", aipp_config, "aipp_24", "success"))["params"]))
    except RuntimeError as e:
        pass
    try:
        aipp(*((gen_static_aipp_case((1, 3, 258, 3), (
            1, 1, 258, 25, 4), "uint8", "uint8", "NHWC", "NC1HWC0_C04", aipp_config, "aipp_24", "success"))["params"]))
    except RuntimeError as e:
        pass
    aipp_config_dict21["input_format"] = "RGB16"
    aipp_config = json.dumps(aipp_config_dict21)
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

    aipp_config_dict22 = aipp_config_dict21.copy()
    aipp_config_dict22["csc_switch"] = 1
    del aipp_config_dict22["resize"]
    aipp_config = json.dumps(aipp_config_dict22)
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

    set_current_compile_soc_info("SD3403")

    aipp_config_dict["input_format"] = "RGB8_IR"
    aipp_config_dict["src_image_size_w"] = 2
    aipp_config = json.dumps(aipp_config_dict)
    try:
        aipp(*((gen_static_aipp_case((1, 3, 258, 1), (1, 1, 258, 25, 4), "uint8", "uint8", "NCHW", "NC1HWC0_C04",
                                     aipp_config, "aipp_24", RuntimeError))["params"]))
    except RuntimeError as e:
        pass

    from impl.aipp_comm import Const
    Const.DEFAULT_MATRIX_R0C1_YUV2RGB == 516

def test_import_lib_const(test_arg):
    import importlib
    import sys
    importlib.reload(sys.modules.get("impl.aipp_comm"))


ut_case.add_cust_test_func(test_func=test_import_lib_const)
ut_case.add_cust_test_func(test_func=test_check_aipp_dtype_001)
ut_case.add_cust_test_func(test_func=test_get_spr9_001)
ut_case.add_cust_test_func(test_func=test_get_spr2_spr9_001)
ut_case.add_cust_test_func(test_func=test_set_aipp_default_params_001)
ut_case.add_cust_test_func(test_func=test_aipp_001)
ut_case.add_cust_test_func(test_func=test_set_spr2_spr9_1)

if __name__ == '__main__':
    #ut_case.run("Ascend310")
    ut_case.run()

