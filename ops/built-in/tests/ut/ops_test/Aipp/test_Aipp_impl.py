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
                    "left_padding_size" : 32,
                    "right_padding_size" : 32,
                    "top_padding_size" : 32,
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

def gen_static_aipp_case(input_shape, output_shape,
                         dtype_x, format, case_name_val, expect):
    return {"params": [{"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": format, "format": format},
                       None,
                       {"shape": output_shape, "dtype": dtype_x, "ori_shape": output_shape, "ori_format": format, "format": format},
                       aipp_config],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710"],
                 gen_static_aipp_case((1,3,418,416), (1,1,418,416,32),
                                      "uint8", "NCHW", "aipp_1", "success"))
ut_case.add_case(["Ascend910"],
                 gen_static_aipp_case((1,3,418,416), (1,1,418,416,32),
                                      "uint8", "NCHW", "aipp_1", RuntimeError))

if __name__ == '__main__':
    # ut_case.run("Ascend710")
    ut_case.run()

