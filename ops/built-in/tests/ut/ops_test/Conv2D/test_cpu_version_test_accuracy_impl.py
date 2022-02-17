#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_cpu_version_accuracy(test_arg):
    import tbe
    from tbe import tvm
    from impl.conv2d import conv2d_compute
    import numpy as np


    def _int_ceil_div(num_a, num_b):
        if num_b == 0:
            raise RuntimeError(" division by zero")

        return (num_a + num_b - 1) // num_b


    def convert_NCHW_to_NC1HWC0(mat):
        n, c, h, w = mat.shape
        return mat.reshape(n, _int_ceil_div(c, 16), 16, h*w).transpose((0, 1, 3, 2)).copy()


    def conver_NC1HWCO_to_NCHW(mat):
        mat = mat.transpose((0, 1, 4, 2, 3)).copy()
        n, c1, c0, h, w = mat.shape
        return mat.reshape(n, c1*c0, h, w)


    def conver_Fraz_to_NCHW(mat, ori_shape):  # C1*KH*KW, Cout1, Cout0, C0,
        mat = mat.transpose((1, 2, 0, 3)).copy()    # Cout1, Cout0,  C1*KH*KW, C0
        n, c, h, w = ori_shape
        mat = mat.reshape(n, _int_ceil_div(c, 16), h, w, 16).transpose((0, 1, 4, 2, 3)).copy()  # Cout, C1, H, W, C0
        return mat.reshape(n, _int_ceil_div(c, 16)*16, h, w)


    def _conv_forward_naive(x, w, conv_param, dilate_h, dilate_w):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        S = conv_param['stride']
        P = conv_param['pad']
        HH_dilate = (HH - 1)*dilate_h + 1
        WW_dilate = (WW - 1)*dilate_w + 1
        Ho = 1 + (H + P[0] + P[1] - HH_dilate) // S[0]
        Wo = 1 + (W + P[2] + P[3] - WW_dilate) // S[1]
        x_pad = np.zeros((N, C, H + P[0] + P[1], W + P[2] + P[3]))
        x_pad[:, :, P[0]:P[0] + H, P[2]:P[2] + W] = x
        out = np.zeros((N, F, Ho, Wo), dtype=np.float16)
        import itertools
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    sub_fm = x_pad[:, :, i * S[0]: i * S[0] + HH_dilate: dilate_h, j * S[1]: j * S[1] + WW_dilate: dilate_w]
                    sub_ft = w[f, :, :, :]
                    idx_fm = list(itertools.product(range(N), range(C), range(i * S[0], i * S[0] + HH), range(j * S[1], j * S[1] + WW)))
                    idx_ft = list(itertools.product([f], range(C), range(HH), range(WW)))
                    res = np.sum(sub_fm * sub_ft, axis=(1, 2, 3))
                    out[:, f, i, j] = res
        cache = (x, w, conv_param)
        return out, cache


    def dsl_cpu_test_fp16(case, cpu_index):

        fm_ori_shape = case['fm_ori_shape']
        fm_shape = case['fm_shape']
        filter_ori_shape = case['filter_ori_shape']
        filter_shape = case['filter_shape']   # C1*KH*KW, Cout1, Cout0, C0,
        bias_flag = int(case['bias_flag'])
        pads = case['pads']
        stride_h = case['stride_h']
        stride_w = case['stride_w']
        dilate_h = case['dilate_h']
        dilate_w = case['dilate_w']

        fmap = tvm.placeholder(fm_shape, name="fmap", dtype="float16", attrs={"ori_shape": fm_ori_shape, "format": "NCHW", "ori_format": "NCHW"})
        weight = tvm.placeholder(filter_shape, name="weight", dtype="float16", attrs={"ori_shape": filter_ori_shape, "format": "FRACTAL_Z", "ori_format": "NCHW"})

        if bias_flag:
            bias_tensor = tvm.placeholder([filter_ori_shape[0], ], name='tensor_bias', dtype="float16")
        else:
            bias_tensor = None

        conv_res = conv2d_compute(fmap, weight, bias_tensor, None, None, [1, 1, stride_h, stride_w], pads, [1, 1, dilate_h, dilate_w], offset_x=0)
        if bias_flag:
            tensor_list = [fmap, weight, bias_tensor, conv_res]
        else:
            tensor_list = [fmap, weight, conv_res]

        sch = tvm.create_schedule(conv_res.op)
        cpu_conv = tvm.build(sch, tensor_list, "c", "llvm", name="cpu_conv_" + str(cpu_index))
        ctx = tvm.cpu(0)

        data_x_5d = tvm.nd.array(np.random.uniform(size=fm_shape).astype(np.float16), ctx)
        data_w_5d = tvm.nd.array(np.random.uniform(size=filter_shape).astype(np.float16), ctx)
        data_x_4d = conver_NC1HWCO_to_NCHW(data_x_5d.asnumpy())
        data_w_4d = conver_Fraz_to_NCHW(data_w_5d.asnumpy(), filter_ori_shape)

        conv_param = {'stride': [stride_h, stride_w], 'pad': pads}
        conv_out_np, _ = _conv_forward_naive(data_x_4d, data_w_4d, conv_param, dilate_h, dilate_w)
        if bias_flag:
            data_bias = tvm.nd.array(np.random.uniform(size=[filter_ori_shape[0], ]).astype(np.float16), ctx)
            data_bias_ = data_bias.asnumpy()
            conv_out_np = conv_out_np + data_bias_[np.newaxis, :, np.newaxis, np.newaxis]

        conv_out_np_5d = convert_NCHW_to_NC1HWC0(conv_out_np)

        output_shape = [int(i) for i in conv_res.shape]
        conv_out_cpu_5d = tvm.nd.array(np.zeros(output_shape, dtype=np.float16), ctx)
        if bias_flag:
            cpu_conv(data_x_5d, data_w_5d, data_bias, conv_out_cpu_5d)
        else:
            cpu_conv(data_x_5d, data_w_5d, conv_out_cpu_5d)

        tvm.testing.assert_allclose(conv_out_np_5d, conv_out_cpu_5d.asnumpy(), 1e-3, 1e-3)
        print("run Conv2D cpu version test accuracy case success")
        # print(conv_out_np_5d)
        # print(conv_out_cpu_5d)
    testcases = [
        {'fm_ori_shape': (1, 32, 28, 28), 'fm_shape': (1, 2, 28, 28, 16),
         'filter_ori_shape': (32, 32, 3, 3), 'filter_shape': (18, 2, 16, 16),
         'stride_h': 1, 'stride_w': 1, 'dilate_h': 1, 'dilate_w': 1,
         'bias_flag': 1, 'pads': [0, 0, 0, 0]},

        {'fm_ori_shape': (1, 32, 28, 28), 'fm_shape': (1, 2, 28, 28, 16),
         'filter_ori_shape': (32, 32, 3, 3), 'filter_shape': (18, 2, 16, 16),
         'stride_h': 1, 'stride_w': 1, 'dilate_h': 2, 'dilate_w': 2,
         'bias_flag': 0, 'pads': [0, 0, 0, 0]},
    ]

    for index, case in enumerate(testcases):
        dsl_cpu_test_fp16(case, index)

print("adding Conv2D cpu version test accuracy case")
ut_case.add_cust_test_func(test_func=test_conv2d_cpu_version_accuracy)
