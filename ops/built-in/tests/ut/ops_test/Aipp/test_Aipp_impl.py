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


if __name__ == '__main__':
    #ut_case.run("Ascend310")
    ut_case.run()

