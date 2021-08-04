#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Aipp Dynamic shape ut case
"""
from op_test_frame.ut import OpUT
import json


ut_case = OpUT("Aipp", "impl.dynamic.aipp", "aipp")


def gen_dynamic_aipp_case(input_shape, params_shape, output_shape, dtype_x, dtype_params, dtype_y, input_format,
                          output_format, in_range, out_range, aipp_config_json, case_name_val, expect):
    return {"params": [{"shape": input_shape, "dtype": dtype_x, "ori_shape": input_shape, "ori_format": input_format,
                        "format": input_format, "range": in_range},
                       {"shape": params_shape, "dtype": dtype_params, "ori_shape": params_shape, "ori_format": "ND",
                        "format": "ND", "range": [[160,3168]]},
                       {"shape": output_shape, "dtype": dtype_y, "ori_shape": output_shape, "ori_format": input_format,
                        "format": output_format, "range": out_range},
                       aipp_config_json],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

aipp_config_dict_dynamic = {"aipp_mode":"dynamic",
                            "related_input_rank":0,
                            "input_format":"YUV420SP_U8",
                            "max_src_image_size":1966400
                            }

aipp_config_json_dynamic = json.dumps(aipp_config_dict_dynamic)
case1 = gen_dynamic_aipp_case((1,3,-1,-1), (3168,), (1,1,-1,-1,16), "uint8", "uint8", "float16", "NCHW", "NC1HWC0",
                              [[1,1],[3,3],[256,256],[256,256]], [[1,1],[1,1],[224,256],[224,256],[16,16]],
                              aipp_config_json_dynamic, "aipp_1", "success")


aipp_config_dict_dynamic["input_format"] = "RGB888_U8"
aipp_config_json_dynamic = json.dumps(aipp_config_dict_dynamic)
case2 = gen_dynamic_aipp_case((1,3,-1,-1), (3168,), (1,1,-1,-1,16), "uint8", "uint8", "uint8", "NCHW", "NC1HWC0",
                              [[1,1],[3,3],[256,256],[256,256]], [[1,1],[1,1],[224,288],[224,288],[32,32]],
                              aipp_config_json_dynamic, "aipp_2", "success")


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
