"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Col2im ut case
"""

import torch
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("Col2im", "impl.col2im", "col2im")

def calc_expect_func(x, output_size,  y, kernel_size, dilation, padding, stride):
    col_n, col_c1, col_w, col_h, col_c0 = x["shape"] # n,c1,hk*wk,ho*wo,c0
    img_n, img_c1, img_h, img_w, img_c0 = y["shape"]
    col_5hd = torch.from_numpy(x["value"])

    col_nckl = col_5hd.permute(0,1,4,2,3).reshape(col_n, col_c1*col_c0*col_w, col_h)
    img_nchw = torch.nn.functional.fold(col_nckl, (img_h, img_w), kernel_size, dilation, (0,0), stride)

    img_5hd = img_nchw.reshape(img_n, img_c1, img_c0, img_h, img_w).permute(0,1,3,4,2)
    return img_5hd.numpy()

ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (3,2,9,64,16), "shape": (3,2,9,64,16),
                "param_type": "input","range_value":[0.1,2.0]},
        {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input","range_value":[10,10]},
        {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (3,2,10,10,16), "shape": (3,2,10,10,16),
                "param_type": "output"}, 
        (3,3), (1,1), (0,0), (1,1)
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1,1,9,256,16), "shape": (1,1,9,256,16),
                "param_type": "input","range_value":[0.1,2.0]},
        {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input","range_value":[18,18]},
        {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1,1,18,18,16), "shape": (1,1,18,18,16),
                "param_type": "output"}, 
        (3,3), (1,1), (0,0), (1,1)
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (3,2,9,64,16), "shape": (3,2,9,64,16),
                "param_type": "input","range_value":[0.1,2.0]},
        {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input","range_value":[10,10]},
        {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (3,2,10,10,16), "shape": (3,2,10,10,16),
                "param_type": "output"}, 
        (3,3), (1,1), (0,0), (1,1)
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [
        {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1,1,9,256,16), "shape": (1,1,9,256,16),
                "param_type": "input","range_value":[0.1,2.0]},
        {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input","range_value":[18,18]},
        {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0", "ori_shape": (1,1,18,18,16), "shape": (1,1,18,18,16),
                "param_type": "output"}, 
        (3,3), (1,1), (0,0), (1,1)
    ],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})