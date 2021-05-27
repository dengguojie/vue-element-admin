#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import conv2D_ut_testcase as tc
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

TYPE_2_NP_TYPE = {
    "float16": np.float16,
    "int8": np.int8,
    "int32": np.int32,
    "int16": np.int16,
}

def _FracZ_to_NCHW(tensor, ci1, Hk, Hw):
    Ftemp = np.shape(tensor)
    print(Ftemp)
    weight = tensor.reshape(ci1, Hk, Hk, Ftemp[1], Ftemp[2], Ftemp[3]).transpose(3,4,0,5,1,2)
    weight = weight.reshape(Ftemp[1]*Ftemp[2], ci1*Ftemp[3], Hk, Hw)
    return weight


def _NC1HWC0_to_NCHW(tensor):
    """
    input tensor is a 5D feature map,
    with a shape [N, C1, H, W, C0]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C, H, W]
    """
    Ftemp = np.shape(tensor)
    F = [Ftemp[0], Ftemp[1]*Ftemp[4], Ftemp[2], Ftemp[3]]
    outputData = np.zeros(F)
    for i in range(Ftemp[0]):
        for j in range(Ftemp[1]):
            for k in range(Ftemp[4]):
                outputData[i,j*Ftemp[4]+k,:,:] = tensor[i,j,:,:,k]
    return outputData

def _NCHW_to_NC1HWC0(tensor):
    """
    input tensor is a 4D feature map,
    with a shape [N, C, H, W]
    padding C to C1*C0, where C0 = 16
    output: tensor_pad[N, C1, H, W, C0]
    """
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, : :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims).transpose(0, 1, 3, 4, 2)

    return tensor_pad

def calc_expect_func(inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x):
    pad_value = offset_x
    if inputs['dtype'] == 'float16':
        c0 = 16
    else:
        c0 = 32
    N, C, H, W = inputs.get('ori_shape')
    F, _, HH, WW = weights.get('ori_shape')
    S = strides[2:4]
    P = pads
    Ho = 1 + (H + P[0] + P[1] - HH) // S[0]
    Wo = 1 + (W + P[2] + P[3] - WW) // S[1]

    x_pad = np.zeros((N, C, H + P[0] + P[1], W + P[2] + P[3]))
    x_pad[:, :, :, :] = pad_value
    x_pad[:, :, P[0]:P[0] + H, P[2]:P[2] + W] = _NC1HWC0_to_NCHW(inputs["value"])
    weights_val = _FracZ_to_NCHW(weights["value"], C//c0, HH, WW)
    mad_type = "int32"
    if outputs["dtype"] is not None:
        mad_type = outputs['dtype']

    out = np.zeros((N, F, Ho, Wo), dtype=TYPE_2_NP_TYPE[mad_type])
    import itertools
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                sub_fm = x_pad[:, :, i * S[0]: i * S[0] + HH, j * S[1]: j * S[1] + WW]
                sub_ft = weights_val[f, :, :, :]
                idx_fm = list(itertools.product(range(N), range(C), range(i * S[0], i * S[0] + HH), range(j * S[1], j * S[1] + WW)))
                idx_ft = list(itertools.product([f], range(C), range(HH), range(WW)))
                res = np.sum(sub_fm * sub_ft, axis=(1, 2, 3))
                out[:, f, i, j] = res
    out = _NCHW_to_NC1HWC0(out)
    return out

def gen_trans_precision_data_case(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,
                                  groups, data_format, offset_x, expect):
    return {
        "params":
            [inputs, weights, bias, offset_w, outputs, strides, pads, dilations, groups, data_format, offset_x],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}


print("adding Conv2D op precision testcases")
for test_case in tc.conv2D_ut_precision_testcase:
    ut_case.add_precision_case(test_case[0], gen_trans_precision_data_case(*test_case[1:]))


if __name__ == '__main__':
    ut_case.run(["Ascend310"], simulator_mode="pv",
                simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
    exit(0)
