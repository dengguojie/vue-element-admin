#!/usr/bin/env python
# -*- coding:utf-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("AvgPoolV2", "impl.dynamic.avg_pool_v2", "avg_pool_v2")

avgpoolv2_ut_testcases = [
    # fmap_shape, filter_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, range, expect_result
    # custom cases H/W dimensions dynamic
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], "success"),

    ((1, 32, -1, -1), (32, 1, 7, 7), (1, 1, 7, 7), (1, 1, 2, 2), "CALCULATED", (1, 1, 2, 2), "NCHW", True, False, 
    [(1,1), (32,32), (1,500), (1,50)], "success"),
    ((1, 32, -1, -1), (32, 1, 8, 8), (1, 1, 8, 8), (1, 1, 1, 1), "CALCULATED", (4, 5, 6, 7), "NCHW", True, False, 
    [(1,1), (32,32), (1,70), (50,200)], "success"),
    ((1, 32, -1, -1), (32, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,85)], "success"),
    # N/H/W dimensions dynamic
    ((-1, 32, -1, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (1,100), (56,56)], "success"),
    ((-1, 32, 100, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (100,100), (1,200)], "success"),
    ((-1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (1,100), (1,100)], "success"),
    # batch dynamic, h/w static
    ((-1, 32, 100, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (32,32), (100,100), (56,56)], "success"),
    ((-1, 128, 768, 680), (128, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,10), (128,128), (768,768), (680,680)], "success"),
    ((-1, 512, 1024, 1024), (512, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "CALCULATED", (2, 2, 2, 2), "NCHW", True, False, 
    [(1,10), (512,512), (1024,1024), (1024,1024)], "success"),

    # load2d
    ((-1, 512, 1024, 1024), (32, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), "CALCULATED", (0,0,0,0), "NCHW", False, False, 
    [(1,10), (512,512), (1024,1024), (1024,1024)], "success"),

    # range [1, None]
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,None), (1,None)], "success"),

    # static shape, need at least one dimension in N/H/W is a variable
     ((1, 32,56, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "VALID", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (56,56), (56,56)], RuntimeError),

    # channel = -1
    ((1, -1, 56, 56), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (56,56), (56,56)], RuntimeError),
    # invalid ksize/strides/pads
    ((1, 32, -1, -1), (32, 1, 3, 3), (3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (2, 2, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (3, 3, 3, 3), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, -1, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid padding_mode
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "EXCUTE", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid format
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "HWCN", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

    # invalid strides
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 3, 3), (1, 1, 64, 64), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid ksize
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 256, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),
    # invalid pads
    ((1, 32, -1, -1), (32, 1, 3, 3), (1, 1, 301, 3), (1, 1, 2, 2), "CALCULATED", (300, 0, 0, 0), "NCHW", True, False, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

    # kuaishou network
    # blocky
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (96,896), (72,672)], "success"),
    # blur
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (512,1200), (512,1200)], "success"),
    # defocus
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (256,512), (256,512)], "success"),
    # noise
    ((16, 3, -1, -1), (3, 1, 3, 3), (1, 1, 3, 3), (1, 1, 1, 1), "CALCULATED", (1, 1, 1, 1), "NCHW", False, False, 
    [(16,16), (3,3), (96,448), (72,448)], "success"),


    # vector
    ((1, 32, -1, -1), None, (1, 1, 3, 3), (1, 1, 2, 2), "SAME", (0, 0, 0, 0), "NCHW", True, True, 
    [(1,1), (32,32), (1,100), (1,100)], RuntimeError),

]

def get_kernel_name(fmap_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range):

    batch_range, c_range, h_range, w_range = in_range
    fmap_info = "_".join([str(i) for i in fmap_shape])
    ksize_info = "_".join([str(i) for i in ksize])
    strides_info = "_".join([str(i) for i in strides])
    pads_info = "_".join([str(i) for i in pads])
    range_info = "_".join([str(batch_range[0]), str(batch_range[1]), str(c_range[0]), str(c_range[1]),
                        str(h_range[0]), str(h_range[1]), str(w_range[0]), str(w_range[1])])

    kernel_name = "fmap_{}_k_{}_s_{}_padding_{}_pads_{}_format_{}_ceilmode_{}_exclusive_{}_range_{}".format(fmap_info,
                    ksize_info, strides_info, padding, pads_info, data_format, str(ceil_mode), str(exclusive), range_info)

    return kernel_name


def gen_avgpoolv2_params(case):
    fmap_shape, filter_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range, expect_result = case

    kernel_name = get_kernel_name(fmap_shape, ksize, strides, padding, pads, data_format, ceil_mode, exclusive, in_range)

    output_shape = fmap_shape
    if filter_shape:
        filter_range = [(filter_shape[0], filter_shape[0]), (filter_shape[1], filter_shape[1]), \
                        (filter_shape[2], filter_shape[2]), (filter_shape[3], filter_shape[3])]
        weight = {"ori_shape": filter_shape, "dtype": "float16", "ori_format": data_format, "range": filter_range}
    else:
        weight = None

    inputs = {"ori_shape": fmap_shape, "dtype": "float16", "ori_format": data_format, "range": in_range}
    outputs = {"ori_shape": output_shape, "dtype": "float16", "ori_format": data_format}

    global_pooling = False
    op_params = [inputs, weight, outputs, ksize, strides, padding, pads, data_format, global_pooling, \
                ceil_mode, exclusive]

    return {"params": op_params,
            "expect": expect_result,
            "support_expect": True}

for testcase in avgpoolv2_ut_testcases:
    ut_case.add_case(["Ascend910A"], gen_avgpoolv2_params(testcase))
    ut_case.add_case(["Ascend710"], gen_avgpoolv2_params(testcase))
    ut_case.add_case(["Ascend310"], gen_avgpoolv2_params(testcase))


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    ut_case.run("Ascend710")
    ut_case.run("Ascend310")
    exit(0)