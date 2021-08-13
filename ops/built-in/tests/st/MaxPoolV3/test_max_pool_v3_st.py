# # -*- coding:utf-8 -*-
import sys
from op_test_frame.ut import BroadcastOpUT
import numpy as np


ut_case = BroadcastOpUT("MaxPoolV3", "impl.max_pool_v3", "max_pool_v3")


def data4Dto5D(input_data, C0=16):
    # input_data = input_data.get("value")
    f_temp = np.shape(input_data)
    f = [f_temp[0], np.int(np.ceil(f_temp[1] * 1.0 / C0)), f_temp[2], f_temp[3], C0]
    output_data = np.zeros(f)
    for N in range(f[0]):
        for C1 in range(f[1]):
            for k in range(C0):
                output_data[N, C1, :, :, k] = input_data[N, C1 * C0 + k, :, :]
    return output_data


def data5Dto4D(input_data):
    input_data = input_data.get("value")
    f_temp = np.shape(input_data)
    f = [f_temp[0], f_temp[1] * f_temp[4], f_temp[2], f_temp[3]]
    output_data = np.zeros(f)
    for i in range(f_temp[0]):
        for j in range(f_temp[1]):
            for k in range(f_temp[4]):
                output_data[i, j * f_temp[4] + k, :, :] = input_data[i, j, :, :, k]
    return output_data


# coding expect function here
def max_pooling_forward(x, y, ksize, strides, padding_mode="CALCULATED", pads=(0, 0, 0, 0),
                        data_format="NCHW", global_pooling=False, ceil_mode=False, kernel_name="max_pool_v3"):
    # def max_pooling_forward(z, pooling, strides=(2, 2), padding=(0, 0), global_pooling=False, ceil_mode=False):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C1, H, W, C0 = x.get('shape')
    C = C1 * C0
    x = data5Dto4D(x)
    if data_format == "NHWC":
        h_index, w_index = 1, 2
    else:
        h_index, w_index = 2, 3
    pooling = (ksize[h_index], ksize[w_index])
    strides = (strides[h_index], strides[w_index])

    if global_pooling:
        pads = (0, 0, 0, 0)
        pooling = (H, W)
    # 零填充
    padding_z = np.lib.pad(x, ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3])), 'constant',
                           constant_values=0)
    # 输出的高度和宽度
    if ceil_mode:
        out_h = (H + pads[0] + pads[1] - pooling[0] + strides[0] - 1) // strides[0] + 1
        out_w = (W + pads[2] + pads[3] - pooling[1] + strides[1] - 1) // strides[1] + 1
    else:
        out_h = (H + pads[0] + pads[1] - pooling[0]) // strides[0] + 1
        out_w = (W + pads[2] + pads[3] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w), dtype=np.float16)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    pool_z = data4Dto5D(pool_z)
    y_dtype = y.get('dtype')
    res = pool_z.astype(y_dtype)
    return [pool_z, ]
