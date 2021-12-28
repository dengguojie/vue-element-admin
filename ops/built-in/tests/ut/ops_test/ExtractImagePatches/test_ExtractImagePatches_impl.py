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

ExtractImagePatches ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.extract_image_patches import extract_image_patches

ut_case = OpUT("ExtractImagePatches", "impl.extract_image_patches", "extract_image_patches")

case1 = {
    "params": [{
        "shape": (1, 2, 4, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 2, 4, 1),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 2, 4, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 2, 4, 1),
        "ori_format": "NHWC"
    }, (1, 2, 2, 1), (1, 3, 3, 1), (1, 3, 3, 1), "SAME"],
    "expect": ValueError,
    "format_expect": [],
    "support_expect": True
}
case2 = {
    "params": [{
        "shape": (1, 2, 10, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 2, 10, 1),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 2, 10, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 2, 10, 1),
        "ori_format": "NHWC"
    }, (1, 4, 4, 1), (1, 3, 3, 1), (1, 3, 3, 1), "SAME"],
    "expect": ValueError,
    "format_expect": [],
    "support_expect": True
}
case3 = {
    "params": [{
        "shape": (2, 16, 1, 64),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (2, 16, 1, 64),
        "ori_format": "NHWC"
    }, {
        "shape": (2, 8, 1, 576),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (2, 8, 1, 576),
        "ori_format": "NHWC"
    }, (1, 3, 3, 1), (1, 2, 2, 1), (1, 3, 3, 1), "SAME"],
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case4 = {
    "params": [{
        "shape": (2, 16, 1, 64),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (2, 16, 1, 64),
        "ori_format": "NHWC"
    }, {
        "shape": (2, 4, 1, 256),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (2, 4, 1, 256),
        "ori_format": "NHWC"
    }, (1, 2, 2, 1), (1, 4, 4, 1), (1, 3, 3, 1), "SAME"],
    "expect": RuntimeError,
    "format_expect": [],
    "support_expect": True
}
case5 = {
    "params": [{
        "shape": (1, 319, 319, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 319, 319, 16),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 40, 40, 258064),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 40, 40, 258064),
        "ori_format": "NHWC"
    }, (1, 127, 127, 1), (1, 8, 8, 1), (1, 1, 1, 1), "SAME"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case6 = {
    "params": [{
        "shape": (1, 319, 319, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 319, 319, 1),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 40, 40, 16129),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 40, 40, 16129),
        "ori_format": "NHWC"
    }, (1, 127, 127, 1), (1, 8, 8, 1), (1, 1, 1, 1), "SAME"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case7 = {
    "params": [{
        "shape": (68, 60, 104, 92),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (68, 60, 104, 92),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 40, 40, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 40, 40, 16),
        "ori_format": "NHWC"
    }, (1, 3, 1, 1), (1, 1, 2, 1), (1, 1, 2, 1), "VALID"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case8 = {
    "params": [{
        "shape": (1, 80, 80, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 80, 80, 16),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 10, 10, 256),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 10, 10, 256),
        "ori_format": "NHWC"
    }, (1, 4, 4, 1), (1, 8, 8, 1), (1, 1, 1, 1), "VALID"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case9 = {
    "params": [{
        "shape": (1, 80, 80, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 80, 80, 1),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 10, 10, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 10, 10, 16),
        "ori_format": "NHWC"
    }, (1, 4, 4, 1), (1, 8, 8, 1), (1, 1, 1, 1), "VALID"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

case10 = {
    "params": [{
        "shape": (2, 154, 441, 1),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (2, 154, 441, 1),
        "ori_format": "NHWC"
    }, {
        "shape": (1, 10, 10, 16),
        "dtype": "float16",
        "format": "NHWC",
        "ori_shape": (1, 10, 10, 16),
        "ori_format": "NHWC"
    }, (1, 1, 4, 1), (1, 25, 26, 1), (1, 3, 1, 1), "VALID"],
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend710", "Ascend910A"], case10)


def extract_image_patches_produce(in_x, conv_param, src_type):
    N, C, H, W = in_x.shape
    HH, WW = conv_param['ksizes']
    SH, SW = conv_param['strides']
    PH, PW = conv_param['pads']
    RH, RW = conv_param['rates']
    if (H + 2 * PH >= (HH - 1) * RH + 1) and (W + 2 * WW >= (HH - 1) * RW + 1):
        Ho = 1 + (H + 2 * PH - ((HH - 1) * RH + 1)) // SH
        Wo = 1 + (W + 2 * PW - ((WW - 1) * RW + 1)) // SW
    else:
        return 0, 0

    if src_type == "fp16" or src_type == "float16":
        s_type = np.float16
        C0 = 16
    elif src_type == "int8":
        s_type = np.int8
        C0 = 32
    elif src_type == "uint8":
        s_type = np.uint8
        C0 = 32
    else:
        raise RuntimeError("unsupported dtype:%s " % src_type)

    Co = HH * WW * C

    C1 = (C + C0 - 1) // C0
    Co1 = (Co + C0 - 1) // C0
    x_pad = np.zeros((N, C1 * C0, H + 2 * PH, W + 2 * PW))
    x_pad[:, :C, PH:PH + H, PW:PW + W] = in_x

    C1 = (C + C0 - 1) // C0
    x_pad = x_pad.reshape(N, C1, C0, H + 2 * PH, W + 2 * PW).transpose(0, 1, 3, 4, 2)
    HoWo = Ho * Wo

    out = np.zeros((N, HoWo, C1, HH * WW, C0))
    print(out.shape)
    for hw in range(HoWo):
        for khkw in range(HH * WW):
            kh = (khkw // WW) * RH
            kw = (khkw % WW) * RW
            y = (hw // Wo) * SH + kh
            x = (hw % Wo) * SW + kw
            out[:, hw, :, khkw, :] = x_pad[:, :, y, x, :]

    out_5HD = np.zeros((N, Co1, HoWo, C0))
    for co1 in range(Co1):
        for co0 in range(C0):
            co = co1 * C0 + co0
            if co < Co:
                ci = co // WW // HH
                ci1 = ci // C0
                ci0 = ci % C0
                khkw = co - ci * WW * HH
                out_5HD[:, co1, :, co0] = out[:, :, ci1, khkw, ci0]
    out_5HD = out_5HD.reshape(N, Co1, Ho, Wo, C0)
    out = out_5HD.transpose(0, 1, 4, 2, 3).reshape(N, -1, Ho, Wo)[:, :Co, :, :]
    c_out = out.reshape(N, C, Co // C, Ho, Wo).transpose(0, 2, 1, 3, 4).reshape(N, Co, Ho, Wo)

    tr_out = c_out.transpose(0, 2, 3, 1)
    return out, tr_out


def get_images(fm_shape, src_type):
    if src_type == "fp16" or src_type == "float16":
        s_type = np.float16
        C0 = 16
    elif src_type == "int8":
        s_type = np.int8
        C0 = 32
    elif src_type == "uint8":
        s_type = np.uint8
        C0 = 32
    else:
        raise RuntimeError("unsupported dtype:%s " % src_type)

    IN, IH, IW, IC = fm_shape
    IC1 = (IC + C0 - 1) // C0

    x = np.array([[[[0 * IC * IH * IW + 0 * IW * IH + h * IW + w + 1 for w in range(IW)] for h in range(IH)]
                   for c in range(IC)] for n in range(IN)])
    ''' transpose to 5D - NC1HWC0 '''
    x_pad = np.zeros((IN, IC1 * C0, IH, IW))
    x_pad[:, :IC, :, :] = x
    feature = x_pad.reshape(IN, IC1, C0, IH, IW).transpose(0, 1, 3, 4, 2).copy()
    return feature.astype(s_type)


def calc_expect_func(images, y, ksizes, strides, dilates, padding):
    feature = images["value"]
    IN, IC1, IH, IW, C0 = feature.shape
    x_pad = feature.transpose(0, 1, 4, 2, 3).reshape(IN, IC1 * C0, IH, IW)
    IN, IH, IW, IC = images["ori_shape"]
    x = x_pad[:, :IC, :, :]
    conv_param = {'ksizes': ksizes[1:3], 'pads': (0, 0), 'strides': strides[1:3], 'rates': dilates[1:3]}
    out, tr_out = extract_image_patches_produce(x, conv_param, images["dtype"])
    return [tr_out]


ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{
            "shape": (2, 7, 7, 32),
            "dtype": "int8",
            "format": "NHWC",
            "ori_shape": (2, 7, 7, 32),
            "ori_format": "NHWC",
            "param_type": "input",
            "value": get_images((2, 7, 7, 32), "int8")
        }, {
            "shape": (2, 3, 3, 288),
            "dtype": "int8",
            "format": "NHWC",
            "ori_shape": (2, 3, 3, 288),
            "ori_format": "NHWC",
            "param_type": "output"
        }, (1, 3, 3, 1), (1, 1, 1, 1), (1, 2, 2, 1), "VALID"],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{
            "shape": (1, 17, 17, 16),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 17, 17, 16),
            "ori_format": "NHWC",
            "param_type": "input",
            "value": get_images((1, 17, 17, 16), "float16")
        }, {
            "shape": (1, 13, 13, 144),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 13, 13, 144),
            "ori_format": "NHWC",
            "param_type": "output"
        }, (1, 3, 3, 1), (1, 1, 1, 1), (1, 2, 2, 1), "VALID"],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{
            "shape": (1, 9, 9, 48),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 9, 9, 48),
            "ori_format": "NHWC",
            "param_type": "input",
            "value": get_images((1, 9, 9, 48), "float16")
        }, {
            "shape": (1, 3, 3, 768),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 3, 3, 768),
            "ori_format": "NHWC",
            "param_type": "output"
        }, (1, 4, 4, 1), (1, 1, 1, 1), (1, 2, 2, 1), "VALID"],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001)
    })

ut_case.add_precision_case(
    "Ascend910A", {
        "params": [{
            "shape": (1, 9, 9, 48),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 9, 9, 48),
            "ori_format": "NHWC",
            "param_type": "input",
            "value": get_images((1, 9, 9, 48), "float16")
        }, {
            "shape": (1, 7, 7, 192),
            "dtype": "float16",
            "format": "NHWC",
            "ori_shape": (1, 7, 7, 192),
            "ori_format": "NHWC",
            "param_type": "output"
        }, (1, 2, 2, 1), (1, 1, 1, 1), (1, 2, 2, 1), "VALID"],
        "calc_expect_func":
            calc_expect_func,
        "precision_standard":
            precision_info.PrecisionStandard(0.001, 0.001)
    })

def test_static_1951(test_arg):   
    set_current_compile_soc_info("Ascend710")
    extract_image_patches({"shape": (1, 16, 16, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 16, 16, 16), "ori_format": "NHWC"},
                          {"shape": (1, 16, 16, 64), "dtype": "float16", "format": "NHWC", "ori_shape": (1, 16, 16, 64), "ori_format": "NHWC"},
                          (1, 2, 2, 1), (1, 1, 1, 1), (1, 3, 3, 1), "SAME")
    set_current_compile_soc_info(test_arg)
ut_case.add_cust_test_func(test_func=test_static_1951)
# if __name__ == '__main__':
#     ut_case.run("Ascend910")
#     exit(0)
