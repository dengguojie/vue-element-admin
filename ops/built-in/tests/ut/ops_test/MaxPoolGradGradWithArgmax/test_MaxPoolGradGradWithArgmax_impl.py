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

MaxPoolGradGradWithArgmax ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("MaxPoolGradGradWithArgmax", None, None)

def case_gen(inputShape, ksize, strides, dtype="float16", pad="VALID", format="NHWC"):
    fmap_n, fmap_h, fmap_w, fmap_c = inputShape
    inputShape = fmap_n, (fmap_c + 15) // 16, fmap_h, fmap_w, 16
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    shape_max_pool_h, pad_top, pad_bottom = \
        _get_windowed_output_size(fmap_h, kernel_h, stride_h, pad)
    shape_max_pool_w, pad_left, pad_right = \
        _get_windowed_output_size(fmap_w, kernel_w, stride_w, pad)
    # shape_max_pool = (fmap_n, shape_max_pool_h, shape_max_pool_w, fmap_c)
    shape_max_pool = (fmap_n, (fmap_c + 15) // 16, shape_max_pool_h, shape_max_pool_w, 16)

    outputShape = shape_max_pool
    argmax_shape = (fmap_n, (fmap_c + 15) // 16, kernel_h*kernel_w, (shape_max_pool[2]*shape_max_pool[3] + 31) // 16 * 16, 16 // 16)
    return inputShape, argmax_shape, outputShape, dtype, ksize, pad, strides, format
def _get_windowed_output_size(input_size, kernel_size,
                              stride, padding_type,
                              dilation_rate=1):
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)

    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got", dilation_rate)

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1  # 3
    if padding_type == "VALID":
        output_size = (input_size - effective_kernel_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = (output_size - 1)*stride + effective_kernel_size - input_size
        # padding_needed = max(0, (output_size - 1) * stride + effective_kernel_size - input_size)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before
    else:
        raise RuntimeError("Unsupported padding type", padding_type)

    return output_size, padding_before, padding_after

def tf_get_windowed_output_size_verbose_V2(input_size, filter_size,
                                           stride, padding_type,
                                           dilation_rate=1):
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

    effective_filter_size = (filter_size - 1) * dilation_rate + 1  # 3
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
    _, fmap_h, fmap_w, _ = input_img.shape  # 5, 5
    stride_h, stride_w = stride  # 1, 1
    pad_top, _, pad_left, _ = pad  # 1, 1

    _, col_ho, col_wo, col_hw, col_ww, _ = col_shape  # 5, 5, 3, 3
    col = np.zeros(col_shape, input_img.dtype)
    for ho in range(col_ho):
        for wo in range(col_wo):
            for hw in range(col_hw):
                for ww in range(col_ww):
                    hi = ho * stride_h + hw - pad_top
                    wi = wo * stride_w + ww - pad_left
                    if hi < 0 or wi < 0 or hi >= fmap_h or wi >= fmap_w:
                        continue
                    col[:, ho, wo, hw, ww, :] = input_img[:, hi, wi, :]

    return col


def check_shape_vailded(ori_input, ori_output, grad, ksize, strides, padding):
    fmap_n, fmap_h, fmap_w, fmap_c = ori_input
    ori_y_n, ori_y_h, ori_y_w, ori_y_c = ori_output
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    output_h, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_h, kernel_h, stride_h,
                                               padding)
    output_w, pad_left, pad_right = \
        tf_get_windowed_output_size_verbose_V2(fmap_w, kernel_w, stride_w,
                                               padding)
    if ori_input != grad:
        raise RuntimeError("ori_input.shape != grad.shape")
    if ori_y_h != output_h or ori_y_w != output_w:
        raise RuntimeError("ori_y_h != output_h or ori_y_w != output_w")
    if fmap_n != ori_y_n:
        raise RuntimeError("fmap_n != ori_y_n")
    if fmap_c != ori_y_c:
        raise RuntimeError("fmap_c != ori_y_c")
    if ksize[0] != 1 or ksize[1] <= 0 or ksize[2] <= 0 or ksize[3] != 1:
        raise RuntimeError(
            "ksize[0] != 1 or ksize[1] <= 0 or ksize[2] <= 0 or ksize[3] != 1")
    if strides[0] != 1 or strides[1] <= 0 or strides[2] <= 0 or strides[3] != 1:
        raise RuntimeError(
            "strides[0] != 1 or strides[1] <= 0 or strides[2] <= 0 or strides[3] != 1")

    pad = (pad_top, pad_bottom, pad_left, pad_right)
    return pad


def maxpool(ori_input, ksize, strides, padding, dtype):
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    fmap_n, fmap_h, fmap_w, fmap_c = ori_input.shape
    # 1, 5, 5, 1
    output_h, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_h, kernel_h, stride_h,
                                               padding)  # 5 3 1 same
    # 3, 0, 0
    output_w, pad_left, pad_right = \
        tf_get_windowed_output_size_verbose_V2(fmap_w, kernel_w, stride_w,
                                               padding)

    col_shape = (fmap_n, output_h, output_w, kernel_h, kernel_w, fmap_c)
    pad = (pad_top, pad_bottom, pad_left, pad_right)

    fmap_x_col = img2col(ori_input, col_shape, pad, (stride_h, stride_w))

    res = np.max(fmap_x_col, axis=(3, 4))
    return res

def maxpoolgradgrad(ori_input, ori_output, grad, ksize, strides, pad):
    dtype = ori_input.dtype
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    output_n, output_h, output_w, output_c = ori_output.shape
    col_shape = (output_n, output_h, output_w, kernel_h, kernel_w, output_c)

    fmap_x_col = img2col(ori_input, col_shape, pad, (stride_h, stride_w))

    esp_min = 2**(-14)
    global_mask = np.zeros(ori_output.shape, dtype)
    mask = np.zeros(fmap_x_col.shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            diff_y = fmap_x_col[:, :, :, h, w, :] - ori_output[:, :, :, :]
            diff_mask = 1 - diff_y / (diff_y + esp_min)

            cur_mask = np.maximum(diff_mask - global_mask, 0)
            global_mask += cur_mask

            mask[:, :, :, h, w, :] = cur_mask
    grad_col = img2col(grad, col_shape, pad, (stride_h, stride_w))
    grad_col_mask = grad_col * mask

    output = np.zeros(ori_output.shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            output += grad_col_mask[:, :, :, h, w, :]
    return output

def maxpoolgradgradwithargmax(x, argmax, dx, ksize, strides, pad):
    dtype = x.dtype
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    col_shape = argmax.shape

    grad_col = img2col(dx, col_shape, pad, (stride_h, stride_w))
    grad_col_mask = grad_col * argmax

    output_shape = []
    output_shape.append(argmax.shape[0])
    output_shape.append(argmax.shape[1])
    output_shape.append(argmax.shape[2])
    output_shape.append(argmax.shape[5])

    output = np.zeros(output_shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            output += grad_col_mask[:, :, :, h, w, :]
    return output


def gen_mask(x, y, ksize, strides, dtype, pad="VALID"):
    dtype = x.dtype
    _, kernel_h, kernel_w, _ = ksize
    _, stride_h, stride_w, _ = strides
    y = maxpool(x, ksize, strides, pad, dtype)
    output_n, output_h, output_w, output_c = y.shape
    col_shape = (output_n, output_h, output_w, kernel_h, kernel_w, output_c)
    esp_min = 1.18e-38
    # esp_min = 2**(-30)
    global_mask = np.zeros(y.shape, dtype)
    pad_list = check_shape_vailded(x.shape, y.shape, x.shape, ksize, strides, pad)
    fmap_x_col = img2col(x, col_shape, pad_list, (stride_h, stride_w))

    mask = np.zeros(fmap_x_col.shape, dtype)
    for h in range(kernel_h):
        for w in range(kernel_w):
            diff_y = fmap_x_col[:, :, :, h, w, :] - y[:, :, :, :]
            diff_mask = 1 - diff_y.astype("float32") / (diff_y.astype("float32") + esp_min)

            cur_mask = np.maximum(diff_mask - global_mask, 0)
            global_mask += cur_mask

            mask[:, :, :, h, w, :] = cur_mask
    return mask

def trans_argmax(argmax):
    n, output_h, output_w, kernel_h, kernel_w, c = argmax.shape
    c1, c0 = c // 16, 16
    howo = output_h*output_w
    howo_16b = (howo + 31) // 16 * 16
    argmax_tmp = argmax.reshape(n, output_h, output_w, kernel_h, kernel_w, c1, c0).transpose(0, 5, 3, 4, 1, 2, 6)
    argmax_tmp = argmax_tmp.reshape(n, c1, kernel_h*kernel_w, howo, c0)


    shape_bitmask_16b = (n*c1*kernel_h*kernel_w*howo_16b)
    bitmask_16b = np.zeros(shape_bitmask_16b, dtype="uint16")
    bitmask_tmp = argmax_tmp.astype("uint16")
    bitmask = np.zeros([n * c1 * kernel_h * kernel_w * output_h * output_w]).astype("uint16")
    for i in range(0, n):
        for j in range(0, c1):
            for m in range(0, kernel_h*kernel_w):
                for nn in range(0, howo):
                    sum = 0
                    for k in range(0, 16):
                        sum += bitmask_tmp[i, j, m, nn, k] * (2**(k%16))
                    bitmask[i*c1*kernel_h*kernel_w*howo + j*kernel_h*kernel_w*howo + m*howo + nn] = sum
                    bitmask_16b[i*c1*kernel_h*kernel_w*howo_16b + j*kernel_h*kernel_w*howo_16b + m*howo_16b + nn] = sum

    # return bitmask
    return bitmask_16b

def trans_bitmask_16b(bitmask_16b, argmax_shape):
    n, output_h, output_w, kernel_h, kernel_w, c = argmax_shape
    c1, c0 = c // 16, 16
    howo = output_h*output_w
    howo_16b = (howo + 31) // 16 * 16
    bitmask_tmp = np.zeros((n, c1, kernel_h*kernel_w, howo, c0)).astype("uint16")

    for i in range(0, n):
        for j in range(0, c1):
            for m in range(0, kernel_h*kernel_w):
                for nn in range(0, howo):
                    sum = bitmask_16b[i*c1*kernel_h*kernel_w*howo_16b + j*kernel_h*kernel_w*howo_16b + m*howo_16b + nn]
                    for k in range(0, 16):
                        bitmask_tmp[i, j, m, nn, k]= sum%2
                        sum = sum//2

    argmax = bitmask_tmp.reshape((n, c1, kernel_h, kernel_w, output_h, output_w, c0)).transpose(0, 4, 5, 2, 3, 1, 6).reshape(n, output_h, output_w, kernel_h, kernel_w, c)
    return argmax

def calc_expect_func(x,
                     grad,
                     argmax,
                     y,
                     ksize,
                     strides,
                     padding):
    n, c1, h, w, c0 = x["shape"]
    x = x["value"].transpose((0,2,3,1,4)).reshape((n, h, w, c1*c0))
    dx = grad["value"].transpose((0,2,3,1,4)).reshape((n, h, w, c1*c0))
    argmax = trans_bitmask_16b(argmax["value"], argmax["ori_shape"])

    pad_list = check_shape_vailded(x.shape, x.shape, dx.shape, ksize, strides,
                                   pad)
    output_maxpoolgradgradwithargmax = maxpoolgradgradwithargmax(x, argmax, dx, ksize, strides, pad_list)
    output_maxpoolgradgradwithargmax_5d = output_maxpoolgradgradwithargmax.reshape(output_maxpoolgradgradwithargmax.shape[0], output_maxpoolgradgradwithargmax.shape[1],
                                                                                   output_maxpoolgradgradwithargmax.shape[2], output_maxpoolgradgradwithargmax.shape[3] // 16, 16)
    output_maxpoolgradgradwithargmax_5d = output_maxpoolgradgradwithargmax_5d.transpose(0, 3, 1, 2, 4)
    return output_maxpoolgradgradwithargmax_5d

def get_argmax(x, ksize, strides, pad, dtype):
    n, c1, h, w, c0 = x.shape
    x = x.transpose((0,2,3,1,4)).reshape((n, h, w, c1*c0))
    y = maxpool(x, ksize, strides, pad, dtype)
    argmax = gen_mask(x, y, ksize, strides, dtype, pad)
    bitmask = trans_argmax(argmax)
    return argmax.shape, bitmask

ori_x = np.random.uniform(1.0, 2.0, (1, 1, 5, 5, 16)).astype("float16")
argmax_shape, bitmask = get_argmax(ori_x, (1, 3, 3, 1), (1, 1, 1, 1), "SAME","float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":ori_x},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input"},
               {"shape": (1, 1, 9, 3, 16), "dtype": "uint16", "ori_shape": argmax_shape, "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":bitmask},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "output"},
               (1, 3, 3, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ori_x = np.random.uniform(1.0, 2.0, (1, 1, 5, 5, 16)).astype("float16")
argmax_shape, bitmask  = get_argmax(ori_x, (1, 4, 4, 1), (1, 1, 1, 1), "SAME","float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":ori_x},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input"},
               {"shape": (1, 1, 16, 3, 16), "dtype": "uint16", "ori_shape": argmax_shape, "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":bitmask},
               {"shape": (1, 1, 5, 5, 16), "dtype": "float16", "ori_shape": (1, 5, 5, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "output"},
               (1, 4, 4, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ori_x = np.random.uniform(1.0, 2.0, (1, 1, 35, 35, 16)).astype("float16")
argmax_shape, bitmask = get_argmax(ori_x, (1, 4, 4, 1), (1, 1, 1, 1), "SAME","float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 1, 35, 35, 16), "dtype": "float16", "ori_shape": (1, 35, 35, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":ori_x},
               {"shape": (1, 1, 35, 35, 16), "dtype": "float16", "ori_shape": (1, 35, 35, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input"},
               {"shape": (1, 1, 16, 78, 16), "dtype": "uint16", "ori_shape": argmax_shape, "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":bitmask},
               {"shape": (1, 1, 35, 35, 16), "dtype": "float16", "ori_shape": (1, 35, 35, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "output"},
               (1, 4, 4, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ori_x = np.random.uniform(1.0, 2.0, (1, 1, 77, 77, 16)).astype("float16")
argmax_shape, bitmask = get_argmax(ori_x, (1, 4, 4, 1), (1, 1, 1, 1), "SAME","float16")
ut_case.add_precision_case("all", {
    "params": [{"shape": (1, 1, 77, 77, 16), "dtype": "float16", "ori_shape": (1, 77, 77, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":ori_x},
               {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "ori_shape": (1, 77, 77, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input"},
               {"shape": (1, 1, 16, 372, 16), "dtype": "uint16", "ori_shape": argmax_shape, "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "input", "value":bitmask},
               {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "ori_shape": (1, 77, 77, 16), "format": "NC1HWC0", "ori_format": "NHWC", "param_type": "output"},
               (1, 4, 4, 1), (1, 1, 1, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

inputShape, argmax_shape, outputShape, dtype, ksize, pad, strides, format = case_gen(inputShape=(1, 768, 768, 16), ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), pad="SAME")

x = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
grad = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
argmax = {"shape": argmax_shape, "ori_shape": argmax_shape, "format": format, "dtype": "uint16", "ori_format": format}
y = {"shape": outputShape, "ori_shape": outputShape, "format": format, "dtype": "float16", "ori_format": format}

case1 = {"params":[x, grad, argmax, y, ksize, strides, pad],
         "case_name": "max_pool_grad_grad_with_argmax_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params":[x, grad, argmax, y, (3,3,3,1), strides, pad],
         "case_name": "max_pool_grad_grad_with_argmax_2",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case3 = {"params":[x, grad, argmax, y, ksize, (3,1,1,1), pad],
         "case_name": "max_pool_grad_grad_with_argmax_3",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

inputShape, argmax_shape, outputShape, dtype, ksize, pad, strides, format = case_gen(inputShape=(8, 71, 339, 16*125), ksize=(1, 21, 186, 1), strides=(1, 15, 43, 1), pad="VALID")
x = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
grad = {"shape": inputShape, "ori_shape": inputShape, "format": format, "dtype": "float16", "ori_format": format}
argmax = {"shape": argmax_shape, "ori_shape": argmax_shape, "format": format, "dtype": "uint16", "ori_format": format}
y = {"shape": outputShape, "ori_shape": outputShape, "format": format, "dtype": "float16", "ori_format": format}
case4 = {"params":[x, grad, argmax, y, ksize, strides, pad],
         "case_name": "max_pool_grad_grad_with_argmax_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"], case4)

if __name__ == '__main__':
    ut_case.run()
    # ut_case.run("Ascend910")
    exit(0)
