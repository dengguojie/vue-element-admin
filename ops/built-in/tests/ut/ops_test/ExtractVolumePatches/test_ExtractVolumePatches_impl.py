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

ExtractVolumePatches ut case
"""
import numpy as np
import random
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("ExtractVolumePatches")

# case_small_shape_same_not_aligned_uint8 = {
#     "params":
#         [
#             {
#                 "shape": (1, 6, 1, 6, 6, 32),
#                 "format": "NDC1HWC0",
#                 "dtype": "uint8",
#                 "ori_shape": (1, 6, 6, 6, 1),
#                 "ori_format": "NDHWC"
#             },
#             {
#                 "shape": (1, 3, 1, 3, 3, 32),
#                 "format": "NDC1HWC0",
#                 "dtype": "uint8",
#                 "ori_shape": (1, 3, 3, 3, 8),
#                 "ori_format": "NDHWC"
#             },
#             (1, 2, 2, 2, 1),
#             (1, 2, 2, 2, 1),
#             "SAME"
#         ],
#     "case_name": 'test_extract_volume_patches_small_shape_same_not_aligned_uint8',
#     "expect": "success"
# }

case_big_shape_same_not_aligned_int8 = {
    "params":
        [
            {
                "shape": (5, 7, 3, 9973, 17, 32),
                "format": "NDC1HWC0",
                "dtype": "int8",
                "ori_shape": (5, 7, 9973, 17, 74),
                "ori_format": "NDHWC"
            },
            {
                "shape": (5, 2, 907, 907, 2, 32),
                "format": "NDC1HWC0",
                "dtype": "int8",
                "ori_shape": (5, 2, 907, 2, 29008),
                "ori_format": "NDHWC"
            },
            (1, 8, 7, 7, 1),
            (1, 5, 11, 11, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_big_shape_same_not_aligned_int8',
    "expect": "success"
}

case_infer_bound_shape_same_not_aligned_int8 = {
    "params":
        [
            {
                "shape": (13, 92, 1, 8, 7, 32),
                "format": "NDC1HWC0",
                "dtype": "int8",
                "ori_shape": (13, 92, 8, 7, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (13, 31, 79056, 1, 1, 32),
                "format": "NDC1HWC0",
                "dtype": "int8",
                "ori_shape": (13, 31, 1, 1, 2529792),
                "ori_format": "NDHWC"
            },
            (1, 488, 18, 18, 1),
            (1, 3, 46, 46, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_infer_bound_shape_same_not_aligned_int8',
    "expect": "success"
}


case_small_shape_same_aligned_fp16_ncdhw = {
    "params":
        [
            {
                "shape": (1, 3, 1, 3, 3, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 16, 3, 3, 3),
                "ori_format": "NCDHW"
            },
            {
                "shape": (1, 3, 8, 3, 3, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 128, 3, 3, 3),
                "ori_format": "NCDHW"
            },
            (1, 1, 2, 2, 2),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_aligned_fp16_ncdhw',
    "expect": "success"
}

case_small_shape_same_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (1, 3, 2, 3, 3, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 3, 3, 3, 17),
                "ori_format": "NDHWC"
            },
            {
                "shape": (1, 3, 9, 3, 3, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 3, 3, 3, 136),
                "ori_format": "NDHWC"
            },
            (1, 2, 2, 2, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_not_aligned_fp16',
    "expect": "success"
}

case_small_shape_same_aligned_multi_batch_fp16 = {
    "params":
        [
            {
                "shape": (32, 2, 1, 2, 2, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (32, 2, 2, 2, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (32, 2, 8, 2, 2, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (32, 2, 2, 2, 128),
                "ori_format": "NDHWC"
            },
            (1, 2, 2, 2, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_small_shape_same_aligned_multi_batch_fp16',
    "expect": "success"
}

case_medium_shape_same_howo_not_aligned_fp16 = {
    "params":
        [
            {
                "shape": (3, 101, 1, 6, 9, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (3, 101, 6, 9, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (3, 51, 50, 2, 5, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (3, 51, 2, 5, 800),
                "ori_format": "NDHWC"
            },
            (1, 2, 5, 5, 1),
            (1, 2, 3, 2, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_medium_shape_same_howo_not_aligned_fp16',
    "expect": "success"
}

case_big_shape_same_howo_aligned_fp16 = {
    "params":
        [
            {
                "shape": (1, 1, 1, 512, 512, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 1, 512, 512, 16),
                "ori_format": "NDHWC"
            },
            {
                "shape": (1, 1, 216, 512, 512, 16),
                "format": "NDC1HWC0",
                "dtype": "float16",
                "ori_shape": (1, 1, 512, 512, 3456),
                "ori_format": "NDHWC"
            },
            (1, 6, 6, 6, 1),
            (1, 1, 1, 1, 1),
            "SAME"
        ],
    "case_name": 'test_extract_volume_patches_big_shape_same_howo_aligned_fp16',
    "expect": "success"
}

# ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_not_aligned_uint8)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_same_not_aligned_int8)
ut_case.add_case(["Ascend910", "Ascend310"], case_infer_bound_shape_same_not_aligned_int8)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_aligned_fp16_ncdhw)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_small_shape_same_aligned_multi_batch_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_medium_shape_same_howo_not_aligned_fp16)
ut_case.add_case(["Ascend910", "Ascend310"], case_big_shape_same_howo_aligned_fp16)

# ut_case.add_case(["Ascend310"], case1)

def tf_get_windowed_output_size_verbose_V2(input_size, filter_size,
                                           stride, padding_type, dilation_rate=1):
    if stride <= 0:
        raise RuntimeError("Stride must be > 0, but got", stride)
    if dilation_rate < 1:
        raise RuntimeError("Dilation rate must be >= 1, but got", dilation_rate)

    effective_filter_size = (filter_size - 1)*dilation_rate + 1
    if padding_type == "VALID":
        output_size = (input_size - effective_filter_size + stride) // stride
        padding_before = 0
        padding_after = 0
    elif padding_type == "SAME":
        output_size = (input_size + stride - 1) // stride
        padding_needed = max(
            0, (output_size - 1)*stride + effective_filter_size - input_size)
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
                    hi = ho*stride_h + hw - pad_top
                    wi = wo*stride_w + ww - pad_left
                    if hi < 0 or wi < 0 or hi >= fmap_h or wi >= fmap_w:
                        continue
                    col[:, ho, wo, hw, ww, :] = input_img[:, hi, wi, :]
    return col

def _extract_volume_patches(ori_input, ksize, strides, padding):
    _, kernel_d, kernel_h, kernel_w, _ = ksize
    _, stride_d, stride_h, stride_w, _ = strides
    fmap_n, fmap_d, fmap_h, fmap_w, fmap_c = ori_input.shape

    output_h, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_h, kernel_h, stride_h, padding)
    output_w, pad_left, pad_right = \
        tf_get_windowed_output_size_verbose_V2(fmap_w, kernel_w, stride_w, padding)

    col_shape = (fmap_n, output_h, output_w, kernel_h, kernel_w, fmap_c)
    pad = (pad_top, pad_bottom, pad_left, pad_right)

    fmap_d_col = np.zeros((fmap_n, fmap_d, output_h, output_w, kernel_h, kernel_w, fmap_c), ori_input.dtype)
    for i in range(fmap_d):
        tmp_input = ori_input[:, i, :, :, :].reshape(fmap_n, fmap_h, fmap_w, fmap_c)
        fmap_x_col = img2col(tmp_input, col_shape, pad, (stride_h, stride_w))
        fmap_d_col[:, i, :, :, :, :, :] = fmap_x_col
    tmp_input = fmap_d_col.reshape(fmap_n, fmap_d, output_h*output_w, kernel_h*kernel_w*fmap_c)
    output_d, pad_top, pad_bottom = \
        tf_get_windowed_output_size_verbose_V2(fmap_d, kernel_d, stride_d, padding)

    col_d_shape = (fmap_n, output_d, output_h*output_w, kernel_d, 1, kernel_h*kernel_w*fmap_c)
    pad = (pad_top, pad_bottom, 0, 0)
    fmap_d_col = img2col(tmp_input, col_d_shape, pad, (stride_d, 1))
    fmap_d_col = fmap_d_col.reshape(fmap_n, output_d, output_h, output_w, kernel_d*kernel_h*kernel_w*fmap_c)
    return fmap_d_col

def get_input(inputShape, src_type):
    if src_type == "fp16" or src_type == "float16":
        s_type = np.float16
        BLOCK_SIZE_ALIGN = 16
    elif src_type == "int8":
        s_type = np.int8
        BLOCK_SIZE_ALIGN = 32
    elif src_type == "uint8":
        s_type = np.uint8
        BLOCK_SIZE_ALIGN = 32
    else:
        raise RuntimeError("unsupported dtype:%s "%src_type)

    inputArr1 = np.zeros(shape=inputShape, dtype=s_type)
    for n in range(inputShape[0]):
        for d in range(inputShape[1]):
            for h in range(inputShape[2]):
                for w in range(inputShape[3]):
                    for c in range(inputShape[4]):
                        inputArr1[n, d, h, w, c] = random.random()*100
    N = inputShape[0]
    D = inputShape[1]
    H = inputShape[2]
    W = inputShape[3]
    C = inputShape[4]

    C1 = (inputShape[4] + BLOCK_SIZE_ALIGN - 1) // BLOCK_SIZE_ALIGN
    C0 = BLOCK_SIZE_ALIGN
    inputArr1_6hd = np.zeros((N, D, H, W, C1 * C0))
    inputArr1_6hd[:,:,:,:,:C] = inputArr1
    inputArr1_6hd = inputArr1_6hd.reshape(N, D, H, W, C1, C0).transpose(0, 1, 4, 2, 3, 5).copy()
    return inputArr1_6hd.astype(s_type)

def calc_expect_func(input_x, output_y, ksizes, strides, padding):
    inputArr1_6hd = input_x["value"]
    N, D, C1, H, W, C0 = inputArr1_6hd.shape
    inputArr1_6hd = inputArr1_6hd.transpose(0, 1, 3, 4, 2, 5).reshape(N, D, H, W, C1*C0)
    Ni, Di, Hi, Wi, Ci = input_x["ori_shape"]
    inputArr1 = inputArr1_6hd[:,:,:,:,:Ci]
    outputArr = _extract_volume_patches(inputArr1, ksizes, strides, padding)
    outputShape = outputArr.shape
    No, Do, Ho, Wo, Co = outputShape
    C1o = (outputShape[4] + C0 - 1) // C0
    C0o = C0
    outputArr_6hd = np.zeros((No, Do, Ho, Wo, C1o * C0o))
    outputArr_6hd[:,:,:,:,:Co] = outputArr
    outputArr_6hd = outputArr_6hd.reshape(No, Do, Ho, Wo, C1o, C0o).transpose(0, 1, 4, 2, 3, 5).copy()
    return outputArr_6hd

# ut_case.add_precision_case("all", {
#     "params": [{"shape": (1,6,6,6,32), "format": "NDC1HWC0", "dtype": "uint8", "ori_shape": (1,6,6,6,32), "ori_format": "NDHWC", "param_type": "input", "value":get_input((1,6,6,6,32), "uint8")},
#                {"shape": (1,3,8,3,3,32), "format": "NDC1HWC0", "dtype": "uint8", "ori_shape": (1,3,3,3,256), "ori_format": "NDHWC", "param_type": "output"},
#                (1, 2, 2, 2, 1), (1, 2, 2, 2, 1), "SAME"],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

# ut_case.add_precision_case("all", {
#     "params": [{"shape": (1,3,3,3,32), "format": "NDC1HWC0", "dtype": "uint8", "ori_shape": (1,3,3,3,32), "ori_format": "NDHWC", "param_type": "input", "value":get_input((1,3,3,3,32), "uint8")},
#                {"shape": (1,2,8,3,3,32), "format": "NDC1HWC0", "dtype": "uint8", "ori_shape": (1,2,3,3,256), "ori_format": "NDHWC", "param_type": "output"},
#                (1, 2, 2, 2, 1), (1, 2, 1, 1, 1), "SAME"],
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
# })

ut_case.add_precision_case("all", {
    "params": [{"shape": (1,3,6,9,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,6,9,16), "ori_format": "NDHWC", "param_type": "input", "value":get_input((1,3,6,9,16), "float16")},
               {"shape": (1,2,50,2,5,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,2,5,800), "ori_format": "NDHWC", "param_type": "output"},
               (1, 2, 5, 5, 1), (1, 2, 3, 2, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("all", {
    "params": [{"shape": (1,3,48,9,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,48,9,16), "ori_format": "NDHWC", "param_type": "input", "value":get_input((1,3,48,9,16), "float16")},
               {"shape": (1,2,40,16,5,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,16,5,640), "ori_format": "NDHWC", "param_type": "output"},
               (1, 2, 4, 5, 1), (1, 2, 3, 2, 1), "SAME"],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
