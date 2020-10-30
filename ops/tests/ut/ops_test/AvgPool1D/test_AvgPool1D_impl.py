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

AvgPool1D ut case
"""
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("AvgPool1D", "impl.avg_pool_1d", "avg_pool_1d")

# TODO fix me run failed so comment
ut_case.add_case(["Ascend910"], {"params": [
    {'shape': (16, 1, 1, 16000, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (16, 1, 1, 16000, 16)},
    {'shape': (16, 1, 1, 16000, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (16, 1, 1, 16000, 16)},
    {'shape': (16, 1, 1, 16000, 16), 'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
     "ori_shape": (16, 1, 1, 16000, 16)},
    4,
    2,
    [1, 1],
    False,
    False],
    "expect": "success",
    "case_name": "test_avg_pool1d_001"})


def avgpool_1d_matrix(shape_input, kernel, stride, pads, ceil_mode, count_include_pad):
    w_in = shape_input[3]
    kernel_size = kernel
    pad_l, pad_r = pads
    stride_size = stride
    w_out = pooling_output_shape(w_in, kernel_size, pad_l, pad_r, stride_size, ceil_mode)
    return cal_shape_div(shape_input, w_in, w_out, kernel_size, pad_l, pad_r,
                         stride_size, ceil_mode, count_include_pad)


def calc_expect_func(x_dict, div_dict, out_dict, ksize, strides, pads, ceil_mode, count_include_pad):
    input_dtype = x_dict["dtype"]
    if input_dtype == "fp16" or input_dtype == "float16":
        pdtype = np.float16
    elif input_dtype == "fp32" or input_dtype == "float32":
        pdtype = np.float32
    elif input_dtype == "int32":
        pdtype = np.int32
    elif input_dtype == "int8":
        pdtype = np.int8
    elif input_dtype == "uint8":
        pdtype = np.uint8
    else:
        raise RuntimeError("unsupported dtype:%s " % input_dtype)

    input_A = x_dict["value"]
    N, C1, H, W, C0 = input_A.shape
    w_in = W

    pad_l, pad_r = pads
    w_in_pad = w_in + pad_l * 2

    if pad_l > 0:
        shape_input = (N, C1, H, w_in_pad, C0)
        mid_input_res = np.zeros(shape_input, dtype=np.float16, order='C')
        for i in range(pad_l):
            mid_input_res[:, :, :, i, :] = 0
        for i in range(w_in):
            mid_input_res[:, :, :, pad_l + i, :] = input_A[:, :, :, i, :]
        for i in range(pad_r):
            mid_input_res[:, :, :, pad_l + w_in + i, :] = 0
        input_A = mid_input_res
    w_out = pooling_output_shape(W, ksize, pad_l, pad_r, strides, ceil_mode)
    shape_out = (N, C1, H, w_out, C0)
    mid_res = np.zeros(shape_out, dtype=np.float16, order='C')
    if count_include_pad == False:
        for w in range(w_out):
            mid_num = np.zeros([N, C1, H, 1, C0])
            for k in range(ksize):
                tmp_num = w * strides + k
                mid_num += input_A[:, :, :, tmp_num:tmp_num + 1, :]
            mid_res[:, :, :, w:w + 1, :] = mid_num
        shape_res = mid_res
        res = shape_res * div_dict["value"]
        return res.astype(pdtype)
    if count_include_pad == True:
        raise RuntimeError("count_include_pad current not support!")


def cal_unit(w_out, w_in, kernel_size, pad_l, pad_r, stride):
    shape = (w_out, 16)
    mid_res = np.zeros(shape, dtype=np.float16, order='C')
    first_data_num = pad_l // stride
    for i in range(first_data_num):
        data_num = 1 / (kernel_size - (pad_l - i * stride))
        mid_res[i] = np.full((16,), data_num)

    last_index = get_last_index(w_out, w_in, kernel_size, pad_l, pad_r, stride)
    mid_data_num = last_index - first_data_num
    last_data_num = w_out - last_index

    for i in range(mid_data_num):
        data_num = 1 / kernel_size
        mid_res[i + first_data_num] = np.full((16,), data_num)
    for i in range(last_data_num):
        data_num = 1 / (kernel_size - ((first_data_num + mid_data_num + i) * stride
                                       + kernel_size - (pad_l + w_in)))
        mid_res[first_data_num + mid_data_num + i] = np.full((16,), data_num)
    return mid_res


def cal_shape_div(shape, w_in, w_out, kernel, pad_l, pad_r,
                  stride, ceil_mode, count_include_pad):
    N, C1, H, W, C0 = shape
    kernel_size = kernel
    if count_include_pad == True:
        if ceil_mode == True:
            pass
        if ceil_mode == False:
            pass
    if count_include_pad == False:
        if ceil_mode == True:
            pass
        if ceil_mode == False:
            unit_res = cal_unit(w_out, w_in, kernel_size, pad_l, pad_r, stride)
            mid_data = np.full((N, C1, H, w_out, C0), unit_res)
    return mid_data


def get_last_index(w_out, w_in, kernel_size, pad_l, pad_r, stride):
    for i in range(w_out):
        res = i * stride + kernel_size
        if res >= w_in + pad_l:
            return i
    return 0


def pooling_output_shape(inputSize, kernelSize, pad_l, pad_r, stride, ceil_mode):
    if ceil_mode == True:
        outputSize = div_rtn((inputSize + pad_l + pad_r
                              - (kernelSize - 1) - 1 + stride - 1), stride) + 1
    else:
        outputSize = div_rtn(inputSize + pad_l + pad_r
                             - (kernelSize - 1) - 1 + 0, stride) + 1
    if pad_l:
        # ensure that the last pooling starts inside the image
        # needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride) >= (inputSize + pad_l):
            outputSize -= 1

    return outputSize


def div_rtn(x, y):
    q = x // y
    r = x % y
    if ((r != 0) and ((r < 0) != (y < 0))):
        q -= 1
    return q


ut_case.add_precision_case("all", {
    "params": [{'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                'ori_shape': (16, 1, 1, 16, 16), 'shape': (16, 1, 1, 16, 16), "param_type": "input"},
               {'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "value": avgpool_1d_matrix((16, 1, 1, 16, 16), 4, 2, [1, 1], False, False).astype("float16"),
                'ori_shape': (16, 1, 1, 8, 16), 'shape': (16, 1, 1, 8, 16), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "ori_shape": (16, 1, 1, 8, 16), "shape": (16, 1, 1, 8, 16), "param_type": "output"},
               4, 2, [1, 1], False, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [{'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                'ori_shape': (16, 1, 1, 160, 16), 'shape': (16, 1, 1, 160, 16), "param_type": "input"},
               {'dtype': "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "value": avgpool_1d_matrix((16, 1, 1, 160, 16), 4, 2, [1, 1], False, False).astype("float16"),
                'ori_shape': (16, 1, 1, 80, 16), 'shape': (16, 1, 1, 80, 16), "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "ori_shape": (16, 1, 1, 80, 16), "shape": (16, 1, 1, 80, 16), "param_type": "output"},
               4, 2, [1, 1], False, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [{'dtype': "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                'ori_shape': (767, 1, 1, 16, 16), 'shape': (767, 1, 1, 16, 16), "param_type": "input"},
               {'dtype': "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "value": avgpool_1d_matrix((767, 1, 1, 16, 16), 4, 2, [1, 1], False, False).astype("float32"),
                'ori_shape': (767, 1, 1, 8, 16), 'shape': (767, 1, 1, 8, 16), "param_type": "input"},
               {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "ori_shape": (767, 1, 1, 8, 16), "shape": (767, 1, 1, 8, 16), "param_type": "output"},
               4, 2, [1, 1], False, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case("all", {
    "params": [{'dtype': "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                'ori_shape': (2, 1, 1, 2, 16), 'shape': (2, 1, 1, 2, 16), "param_type": "input"},
               {'dtype': "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "value": avgpool_1d_matrix((2, 1, 1, 2, 16), 4, 2, [1, 1], False, False).astype("float32"),
                'ori_shape': (2, 1, 1, 1, 16), 'shape': (2, 1, 1, 1, 16), "param_type": "input"},
               {"dtype": "float32", "format": "NC1HWC0", "ori_format": "NC1HWC0",
                "ori_shape": (2, 1, 1, 1, 16), "shape": (2, 1, 1, 1, 16), "param_type": "output"},
               4, 2, [1, 1], False, False],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
