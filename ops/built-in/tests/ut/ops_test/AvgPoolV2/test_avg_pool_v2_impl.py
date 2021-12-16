# # -*- coding:utf-8 -*-
import sys
import numpy as np
import te
from te import tvm
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info


ut_case = BroadcastOpUT("avg_pool_v2")


def NCHW2C1HWNCOC0(x, shape, dtype):
    y = np.zeros(shape, dtype)
    for c1 in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                for n in range(shape[3]):
                    for co in range(shape[4]):
                        for c0 in range(shape[5]):
                            if co == c0:
                                try:
                                    y[c1][h][w][n][co][c0] = x[n][co][h][w]
                                except:
                                    y[c1][h][w][n][co][c0] = 0

    return y


def NC1HWC02NCHW(fmi, fmi_shape, precise, shape_orig):
    if precise=='int8':
        fmo = np.zeros((fmi_shape[0], fmi_shape[1]*fmi_shape[4], fmi_shape[2], fmi_shape[3]), dtype=np.int8)
    else:
        fmo = np.zeros((fmi_shape[0], shape_orig[1], fmi_shape[2], fmi_shape[3]), dtype=np.float16)
    for n in range(fmi_shape[0]):
        for c1 in range(fmi_shape[1]):
            for h in range(fmi_shape[2]):
                for w in range(fmi_shape[3]):
                    for c0 in range(fmi_shape[4]):
                        if c1*fmi_shape[4]+c0 < shape_orig[1]:
                            fmo[n][c1*fmi_shape[4]+c0][h][w] = fmi[n][c1][h][w][c0]
    return fmo


def avg_pool_forward(z, pooling, strides=(1,1), padding_mode="CALCULATED", padding=(0,0,0,0), global_pooling=True, ceil_mode=False, exclusive=True):
    N, C, H, W = z.shape
    if global_pooling or (padding == (0, 0, 0, 0) and pooling == (H, W)):
        padding = (0, 0, 0, 0)
        pooling = (H, W)
        padding_mode = "VALID"

    if padding_mode == "SAME":
        out_h = (H + strides[0] - 1) // strides[0]
        out_w = (W + strides[1] - 1) // strides[1]
        pad_row = (out_h - 1) * strides[0] + ((pooling[0] - 1) + 1) - H
        pad_col = (out_w - 1) * strides[1] + ((pooling[1] - 1) + 1) - W

        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left

        pad_top = max(int(pad_top), 0)
        pad_bottom = max(int(pad_bottom), 0)
        pad_left = max(int(pad_left), 0)
        pad_right = max(int(pad_right), 0)

        padding = (pad_top, pad_bottom, pad_left, pad_right)

    padding_z = np.lib.pad(z, ((0,0), (0,0), (padding[0], padding[1]), (padding[2], padding[3])), 'constant', constant_values=0)

    if padding_mode == "CALCULATED":
        if ceil_mode:
            out_h = (H + padding[0] + padding[1] - pooling[0] + strides[0] - 1) // strides[0] + 1
            out_w = (W + padding[2] + padding[3] - pooling[1] + strides[1] - 1) // strides[1] + 1
        else:
            out_h = (H + padding[0] + padding[1] - pooling[0]) // strides[0] + 1
            out_w = (W + padding[2] + padding[3] - pooling[1]) // strides[1] + 1

    if padding_mode == "VALID":
        out_h = (H - pooling[0] + 1 + strides[0] - 1) // strides[0]
        out_w = (W - pooling[1] + 1 + strides[1] - 1) // strides[1]

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    window = padding_z[n, c, strides[0]*i:strides[0]*i+pooling[0], strides[1]*j:strides[1]*j+pooling[1]]
                    window_size = pooling[0] * pooling[1]
                    if global_pooling or (padding == (0, 0, 0, 0) and pooling == (H, W)):
                        mean = np.mean(window)
                    else:
                        mean = np.mean(window)
                        mean = mean * window_size
                    pool_z[n, c, i, j] = mean
    return pool_z


def NCHW2NC1HWC0(fmi, fmo_shape, precise):
    if precise == "int8":
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.int8)
    else:
        fmo = np.zeros((fmo_shape[0], fmo_shape[1], fmo_shape[2], fmo_shape[3], fmo_shape[4]), dtype=np.float16)
    for n in range(fmo_shape[0]):
        for c1 in range(fmo_shape[1]):
            for h in range(fmo_shape[2]):
                for w in range(fmo_shape[3]):
                    for c0 in range(fmo_shape[4]):
                        try:
                            fmo[n][c1][h][w][c0] = fmi[n][c1*fmo_shape[4]+c0][h][w]
                        except:
                            fmo[n][c1][h][w][c0] =0
    return fmo


# [TODO] coding expect function here
def calc_expect_func(x, filter, y, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode, exclusive):
    train_data = NC1HWC02NCHW(x['value'], x['shape'], "float16", x['ori_shape'])
    outs = avg_pool_forward(train_data, (ksize[-2], ksize[-1]), (strides[-2], strides[-1]), padding_mode,
                            pads, global_pooling, ceil_mode, exclusive)
    res = NCHW2NC1HWC0(outs, y['shape'], "float16")
    return [res, ]


# [TODO] coding cases here
ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,3,3), "shape": (1,1,3,3,16),
                "param_type": "input"},
               {"dtype": "float16", "format": "FRACTAL_Z", "ori_format": "NCHW", "ori_shape": (16,1,2,2), "shape": (64,1,16,16),"param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16,), "shape": (16,),
                "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,1,1), "shape": (1,1,1,1,16),"param_type": "output"},
               [1,1,2,2],
               [1,1,2,2],
               "VALID",
               [0,0,0,0],
               "NCHW",
               False,
               False,
               True],
    "expect": "success",
    "support_expect": True
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "shape": (1, 8, 32, 32, 16), "ori_shape": (1, 32, 32, 128),"param_type": "input"},
               None,
               None,
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (1,16,16,16), "shape": (1,1,16,16,16),"param_type": "output"},
               [1,2,2,1],
               [1,2,2,1],
               "SAME",
               [0,0,0,0],
               "NHWC",
               False,
               False,
               True],
    "expect": "success",
    "support_expect": True})



#ut_case.add_precision_case("Ascend910A", {
    #"params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,3,3), "shape": (1,1,3,3,16),
             #   "param_type": "input"},
             #  {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "NCHW", "ori_shape": (1,16,2,2), "shape": (1,2,2,1,16,16),
             #   "value": NCHW2C1HWNCOC0(np.ones((1,16,2,2), np.float16), (1,2,2,1,16,16), np.float16),
              #  "param_type": "input"},
              # {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,2,2), "shape": (1,1,2,2,16),
              #  "param_type": "output"},
             #  (1,1,2,2),
             #  (1,1,2,2),
          #     "SAME",
         #      (0,0,0,0),
        #       "NCHW",
     #          False,
     #          False,
    #           True],
   # "calc_expect_func": calc_expect_func,
 #   "precision_standard": precision_info.PrecisionStandard(50, 50)
#})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,3,3), "shape": (1,1,3,3,16),
                "param_type": "input"},
               {"dtype": "float16", "format": "C1HWNCoC0", "ori_format": "NCHW", "ori_shape": (1,16,2,2), "shape": (1,2,2,1,16,16),
                "value": NCHW2C1HWNCOC0(np.ones((1,16,2,2), np.float16), (1,2,2,1,16,16), np.float16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16,), "shape": (16,),
                "param_type": "input"},
               {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,2,2), "shape": (1,1,2,2,16),
                "param_type": "output"},
               (1,1,2,2),
               (1,1,2,2),
               "SAME",
               (0,0,0,0),
               "NCHW",
               False,
               False,
               True],
    "expect": ZeroDivisionError
})


from impl.avg_pool_v2 import check_supported
from impl.avg_pool_v2 import avg_pool_v2_compute
def test_check_support(test_arg):
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 24, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED",[0,0,0,0],"NHWC")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 24, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED",[0,0,0,0],"NCHW")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 1, 1, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 1, 1, 256),"ori_format": "ND", "param_type": "output"},
    [1,2,2,1],[1,4,4,1],"VALIED",[0,0,0,0],"NHWC")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 3, 3, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 3, 256),"ori_format": "ND", "param_type": "output"},
    [1,255,21,1],[1,4,4,1],"VALIED",[0,0,0,0],"NHWC")
    check_supported({"shape": (1, 24, 1, 256), "dtype": "float16", "format": "ND", "ori_shape": (1, 24, 1, 256),"ori_format": "ND", "param_type": "input"},
    None,None,{"shape": (1, 3, 3, 256 ), "dtype": "float16", "format": "ND", "ori_shape": (1, 3, 3, 256),"ori_format": "ND", "param_type": "output"},
    [1,255,21,1],[1,4,64,1],"VALIED",[0,0,0,0],"NHWC")

def test_avg_pool_v2_compute(test_arg):
    a=tvm.placeholder((1, 8, 12, 12, 16), name="fmap", dtype="float16", attrs={"ori_shape":(1, 12, 12, 128), "format":"NC1HWC0", "ori_format":"NHWC","shape":(1, 8, 12, 12, 16)})
    out={"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC", "ori_shape": (1,1,1,128), "shape": (1,8,1,1,16)}
    avg_pool_v2_compute(a,None,None,out,[1,12,12,1],[1,1,1,1],"VALID",(0,0,0,0),"NHWC",True,False,True)
                        
                        
    b=tvm.placeholder((1,1,3,3,16), name="fmap", dtype="float16", attrs={"shape":(1,1,3,3,16),"ori_shape":(1,16,3,3), "format":"NC1HWC0", "ori_format":"NCHW"})
    c=tvm.placeholder((4,1,16,16), name="filter", dtype="float16", attrs={"shape":(4,1,16,16),"ori_shape":(16,1,2,2), "format":"FRACTAL_Z", "ori_format":"NCHW"})
    bias1=tvm.placeholder((16,), name="bias", dtype="float16", attrs={"shape":(16,),"ori_shape":(16,), "format":"ND", "ori_format":"ND"})
    out1={"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,1,1), "shape": (1,1,1,1,16)}
    avg_pool_v2_compute(b,c,bias1,out1,[1,1,2,2],[1,1,2,2],"VALID",(0,0,0,0),"NCHW",False,False,True)

    d=tvm.placeholder((1,1,3,3,16), name="fmap", dtype="float16", attrs={"shape":(1,1,3,3,16),"ori_shape":(1,16,3,3), "format":"NC1HWC0", "ori_format":"NCHW"})
    e=tvm.placeholder((4,1,16,16), name="filter", dtype="float16", attrs={"shape":(4,1,16,16),"ori_shape":(16,1,2,2), "format":"FRACTAL_Z", "ori_format":"NCHW"})
    bias2=tvm.placeholder((16,), name="bias", dtype="float16", attrs={"shape":(16,),"ori_shape":(16,), "format":"ND", "ori_format":"ND"})
    out1={"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW", "ori_shape": (1,16,1,1), "shape": (1,1,1,1,16)}
    avg_pool_v2_compute(d,e,bias2,out1,[1,1,2,2],[1,1,2,2],"CALCULATED",(0,0,0,0),"NCHW",False,False,False)
ut_case.add_cust_test_func(test_func=test_check_support)
ut_case.add_cust_test_func(test_func=test_avg_pool_v2_compute)
