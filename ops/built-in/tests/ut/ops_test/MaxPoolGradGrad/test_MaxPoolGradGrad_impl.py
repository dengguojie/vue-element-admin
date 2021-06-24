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

MaxPoolGradGrad ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPoolGradGrad", None, None)


def gen_case(inp, out):
    (N, C1, H, W, C0) = inp
    (Nout, C1out, Hout, Wout, C0out) = out
    C = C1 * C0
    Cout = C1out * C0out
    ori_x_input = {
        "shape": (N, C1, H, W, C0),
        "dtype": "float16",
        "ori_shape": (N, H, W, C),
        "format": "NC1HWC0",
        "ori_format": "NHWC"
    }
    ori_y_input = {
        "shape": (Nout, C1out, Hout, Wout, C0),
        "dtype": "float16",
        "ori_shape": (Nout, Hout, Wout, Cout),
        "format": "NC1HWC0",
        "ori_format": "NHWC"
    }
    grads = ori_x_input
    output = ori_y_input

    return ori_x_input, ori_y_input, grads, output


ksize = (1, 3, 3, 1)
strides = (1, 1, 1, 1)
ori_x_input, ori_y_input, grads, output = gen_case((1, 1, 5, 5, 16), (1, 1, 5, 5, 16))
case1 = {
    "params": [ori_x_input, ori_y_input, grads, output, ksize, strides],
    "case_name": "max_pool_grad_grad_1",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ksize = (1, 2, 3, 1)
strides = (1, 11, 4, 1)
ori_x_input, ori_y_input, grads, output = gen_case((627, 5, 56, 301, 16), (627, 5, 6, 76, 16))
case2 = {
    "params": [ori_x_input, ori_y_input, grads, output, ksize, strides],
    "case_name": "max_pool_grad_grad_2",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)

ksize = (1, 12, 2, 1)
strides = (1, 58, 33, 1)
ori_x_input, ori_y_input, grads, output = gen_case((37, 3, 120, 44, 16), (37, 3, 3, 2, 16))
case3 = {
    "params": [ori_x_input, ori_y_input, grads, output, ksize, strides],
    "case_name": "max_pool_grad_grad_3",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)

ksize = (1, 13, 6, 1)
strides = (1, 6, 29, 1)
ori_x_input, ori_y_input, grads, output = gen_case((89, 5, 93, 151, 16), (89, 5, 16, 6, 16))
case4 = {
    "params": [ori_x_input, ori_y_input, grads, output, ksize, strides],
    "case_name": "max_pool_grad_grad_4",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

ksize = (1, 14, 16, 1)
strides = (1, 63, 40, 1)
ori_x_input, ori_y_input, grads, output = gen_case((21, 13, 48, 89, 16), (21, 13, 1, 3, 16))
case5 = {
    "params": [ori_x_input, ori_y_input, grads, output, ksize, strides],
    "case_name": "max_pool_grad_grad_5",
    "expect": "success",
    "format_expect": [],
    "support_expect": True
}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case5)


def tf_get_windowed_output_size_verbose_V2(input_size, filter_size,
                                           stride, padding_type, dilation_rate=1):
    """
    get output and padding size using tensorflow padding rule

    Parameters
    ----------
    input_size : int, feature map size

    filter_size : int, filter size

    dilation_rate: int, dilation rate

    stride: int, stride size

    padding_type: string, support "SAME", "VALID" or "EXPLICIT"

    Returns
    -------
    output_size: int, output feature map size

    padding_before: int, feature map padding before size

    padding_after: int, feature map padding after size
    """
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)

    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got", dilation_rate)

    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = max(
            0, (output_size - 1) * stride + effective_filter_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after


def img2col(input_img, col_shape, pad, stride, tag=None):
    _, fmap_h, fmap_w, _ = input_img.shape
    stride_h, stride_w = stride
    pad_top, _, pad_left, _ = pad

    _, col_ho, col_wo, col_hw, col_ww, _ = col_shape
    col = np.zeros(col_shape, input_img.dtype)
    for ho in range(col_ho):
        for wo in range(col_wo):
            for hw in range(col_hw):
                for ww in range(col_ww):
                    hi = ho * stride_h + hw - pad_top
                    wi = wo * stride_w + ww - pad_left
                    if hi < 0 or wi < 0 or hi >= fmap_h or wi >= fmap_w:
                        col[:, ho, wo, hw, ww, :] = -65500.0
                        continue
                    col[:, ho, wo, hw, ww, :] = input_img[:, hi, wi, :]

    return col


def check_shape_vailded(ori_input, ori_output, grad, ksize, strides, padding):
    fmap_n, fmap_h, fmap_w, fmap_c = ori_input
    ori_y_n, ori_y_h, ori_y_w, ori_y_c = ori_output
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    output_h, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_h, kernel_h, stride_h, padding)
    output_w, pad_left, pad_right = \
        tf_get_windowed_output_size_verbose_V2(fmap_w, kernel_w, stride_w, padding)
    if ori_input != grad:
        raise RuntimeError("ori_input.shape != grad.shape")
    if ori_y_h != output_h or ori_y_w != output_w:
        print("ori_y_h:", ori_y_h, "ori_y_w:", ori_y_w, "output_h:", output_h, "output_w:", output_w)
        raise RuntimeError("ori_y_h != output_h or ori_y_w != output_w")
    if fmap_n != ori_y_n:
        raise RuntimeError("fmap_n != ori_y_n")
    if fmap_c != ori_y_c:
        raise RuntimeError("fmap_c != ori_y_c")
    if ksize[0] != 1 or ksize[1] <= 0 or ksize[2] <= 0 or ksize[3] != 1:
        raise RuntimeError("ksize[0] != 1 or ksize[1] <= 0 or ksize[2] <= 0 or ksize[3] != 1")
    if strides[0] != 1 or strides[1] <= 0 or strides[2] <= 0 or strides[3] != 1:
        raise RuntimeError("strides[0] != 1 or strides[1] <= 0 or strides[2] <= 0 or strides[3] != 1")

    pad = (pad_top, pad_bottom, pad_left, pad_right)
    return pad


def maxpool(ori_input, ksize, strides, padding, dtype):
    n, c1, h, w, c0 = ori_input.shape
    ori_input = ori_input.transpose((0, 2, 3, 1, 4)).reshape((n, h, w, c1 * c0))
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    fmap_n, fmap_h, fmap_w, fmap_c = ori_input.shape

    output_h, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_h, kernel_h, stride_h, padding)
    output_w, pad_left, pad_right = \
        tf_get_windowed_output_size_verbose_V2(fmap_w, kernel_w, stride_w, padding)

    col_shape = (fmap_n, output_h, output_w, kernel_h, kernel_w, fmap_c)
    pad = (pad_top, pad_bottom, pad_left, pad_right)

    fmap_x_col = img2col(ori_input, col_shape, pad, (stride_h, stride_w))

    res = np.max(fmap_x_col, axis=(3, 4))
    return res.reshape((n, h, w, c1, c0)).transpose((0, 3, 1, 2, 4)).astype(dtype)


# NHWC
def maxpoolgradgrad_nhwc(ori_input, ori_output, grad, ksize, strides, pad):
    dtype = ori_input.dtype
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    output_n, output_h, output_w, output_c = ori_output.shape
    col_shape = (output_n, output_h, output_w, kernel_h, kernel_w, output_c)

    fmap_x_col = img2col(ori_input, col_shape, pad, (stride_h, stride_w))

    esp_min = 1.18e-38
    global_mask = np.zeros(ori_output.shape, dtype)
    ori_output = ori_output.reshape((ori_output.shape[0], ori_output.shape[1],
                                     ori_output.shape[2], 1, 1, ori_output.shape[3]))
    diff_y = fmap_x_col - ori_output
    diff_mask = 1 - diff_y / (diff_y + esp_min)

    mask = np.zeros(fmap_x_col.shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            cur_mask = np.maximum(diff_mask[:, :, :, h, w, :] - global_mask[:, :, :, :], 0)
            global_mask += cur_mask
            mask[:, :, :, h, w, :] = cur_mask

    grad_col = img2col(grad, col_shape, pad, (stride_h, stride_w))
    grad_col_mask = grad_col * mask

    output = np.zeros(global_mask.shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            output += grad_col_mask[:, :, :, h, w, :]
    return output


def calc_expect_func(orig_x_dict, orig_y_dict, grads_dict, output_dict, ksize, strides, padding):
    n, c1, h, w, c0 = orig_x_dict["shape"]
    ori_x = orig_x_dict["value"].transpose((0, 2, 3, 1, 4)).reshape((n, h, w, c1 * c0)).astype(np.float32)
    ori_y = orig_y_dict["value"].transpose((0, 2, 3, 1, 4)).reshape((n, h, w, c1 * c0)).astype(np.float32)

    pad_list = check_shape_vailded(orig_x_dict["ori_shape"], orig_y_dict["ori_shape"], grads_dict["ori_shape"], ksize,
                                   strides, padding)
    grads = grads_dict["value"].transpose((0, 2, 3, 1, 4)).reshape((n, h, w, c1 * c0)).astype(np.float32)
    output = maxpoolgradgrad_nhwc(ori_x, ori_y, grads, ksize, strides, pad_list)
    return output.reshape((n, h, w, c1, c0)).transpose((0, 3, 1, 2, 4)).astype(output_dict["dtype"])


ori_x = np.random.uniform(1.0, 2.0, (1, 1, 5, 5, 16)).astype("float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input", "value": ori_x},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input",
                "value": maxpool(ori_x, (1, 3, 3, 1), (1, 1, 1, 1), "SAME", "float16")},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input"},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "output"},
               (1, 3, 3, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ori_x = np.random.uniform(1.0, 2.0, (2, 2, 5, 5, 16)).astype("float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 2, 5, 5, 16), "dtype": "float16", "ori_shape": (2, 5, 5, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input", "value": ori_x},
               {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "ori_shape": (2, 5, 5, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input",
                "value": maxpool(ori_x, (1, 3, 3, 1), (1, 1, 1, 1), "SAME", "float16")},
               {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "ori_shape": (2, 5, 5, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input"},
               {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "ori_shape": (2, 5, 5, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "output"},
               (1, 3, 3, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ori_x = np.random.uniform(1.0, 2.0, (2, 2, 5, 13, 16)).astype("float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 2, 5, 13, 16), "dtype": "float16", "ori_shape": (2, 5, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input", "value": ori_x},
               {"shape": (2, 2, 5, 13, 16), "dtype": "float16", "ori_shape": (2, 5, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input",
                "value": maxpool(ori_x, (1, 3, 3, 1), (1, 1, 1, 1), "SAME", "float16")},
               {"shape": (2, 2, 5, 13, 16), "dtype": "float16", "ori_shape": (2, 5, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input"},
               {"shape": (2, 2, 5, 13, 16), "dtype": "float16", "ori_shape": (2, 5, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "output"},
               (1, 3, 3, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ori_x = np.random.uniform(1.0, 2.0, (2, 2, 17, 13, 16)).astype("float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (2, 2, 17, 13, 16), "dtype": "float16", "ori_shape": (2, 17, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input", "value": ori_x},
               {"shape": (2, 2, 17, 13, 16), "dtype": "float16", "ori_shape": (2, 17, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input",
                "value": maxpool(ori_x, (1, 3, 3, 1), (1, 1, 1, 1), "SAME", "float16")},
               {"shape": (2, 2, 17, 13, 16), "dtype": "float16", "ori_shape": (2, 17, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "input"},
               {"shape": (2, 2, 17, 13, 16), "dtype": "float16", "ori_shape": (2, 17, 13, 32), "format": "NC1HWC0",
                "ori_format": "NHWC", "param_type": "output"},
               (1, 5, 5, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run('Ascend310')
    ut_case.run('Ascend910')
