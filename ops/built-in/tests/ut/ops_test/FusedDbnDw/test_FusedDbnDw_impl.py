#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
from op_test_frame.ut import OpUT

ut_case = OpUT(
    "fused_dbn_dw",
    "impl.fused_dbn_dw",
    "fused_dbn_dw",
)
DEBUG_MODE = False
BLOCK = 16
fused_dbn_dw_testcase = {
    # fmap, dedy, filter, stride, pad
    ((32, 64, 56, 56), (32, 64, 56, 56), (64, 64, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((32, 256, 56, 56), (32, 64, 56, 56), (64, 256, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((32, 64, 56, 56), (32, 256, 56, 56), (256, 64, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((32, 64, 56, 56), (32, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1)),
    ((32, 256, 56, 56), (32, 512, 28, 28), (512, 256, 1, 1), (2, 2), (0, 0, 0, 0)),
    ((32, 512, 28, 28), (32, 128, 28, 28), (128, 512, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((32, 256, 56, 56), (32, 128, 56, 56), (128, 256, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((32, 3, 224, 224), (32, 64, 112, 112), (64, 3, 7, 7), (2, 2), (2, 3, 2, 3)),
    ((256, 256, 14, 14), (256, 1024, 14, 14), (1024, 256, 1, 1), (1, 1), (0, 0, 0, 0)),
    ((256, 512, 28, 28), (256, 128, 28, 28), (128, 512, 1, 1), (1, 1), (0, 0, 0, 0)),
}



def _gen_kernel_name(fmap_shape, dedy_shape, w_shape, strides):
    dedy_c, dedy_h = dedy_shape[1:3]
    fmap_c, fmap_h = fmap_shape[1:3]
    filter_h = w_shape[2]
    
    if filter_h > 1:
        if strides[0] > 1:
            case_name = "fuse_dbn_dw_case_{}_{}_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c,  fmap_c, filter_h, strides[0])
        else:
            case_name = "fuse_dbn_dw_case_{}_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c,  fmap_c, filter_h)
    else:
        case_name = "fuse_dbn_dw_case_{}_{}_{}_{}".format(dedy_h, fmap_h, dedy_c,  fmap_c)
    return case_name


def shape_4d_to_5hd(shape):
    """ trans data from 4d to NC1HWC0 """
    if len(shape) != 4:
        return shape
    return (
        shape[0],
        (shape[1] + BLOCK - 1) // BLOCK,
        shape[2],
        shape[3],
        BLOCK,
    )


def shape_4d_to_fz(shape):
    """ trans data from 4d to fz """
    c_out, c_in, height, weight = shape
    c_in1 = (c_in + BLOCK - 1) // BLOCK
    c_in0 = BLOCK
    c_out1 = (c_out + BLOCK - 1) // BLOCK
    c_out0 = BLOCK
    return c_in1 * height * weight, c_out1, c_out0, c_in0


def _gen_trans_data_case(
    fmap_shape,
    dedy_shape,
    filter_shape,
    stride,
    padding,
    dilations=(1, 1, 1, 1),
    groups=1,
    expect="success",
    data_flow="default",
):

    kernel_name = _gen_kernel_name(fmap_shape, dedy_shape, filter_shape, stride)
    format = "NC1HWC0"
    ori_format = "NCHW"

    fm_list = {
        "shape": shape_4d_to_5hd(fmap_shape),
        "dtype": "float16",
        "format": format,
        "ori_shape": fmap_shape,
        "ori_format": ori_format,
    }

    dedy_list = {
        "shape": shape_4d_to_5hd(dedy_shape),
        "dtype": "float16",
        "format": format,
        "ori_shape": dedy_shape,
        "ori_format": ori_format,
    }
    
    dbn_x_list = {
        "shape": shape_4d_to_5hd(dedy_shape),
        "dtype": "float16",
        "format": format,
        "ori_shape": dedy_shape,
        "ori_format": ori_format,
    }

    dbn_vector_shape = (1, dedy_shape[1], 1, 1)

    dbn_vector_list = {
        "shape": shape_4d_to_5hd(dbn_vector_shape),
        "dtype": "float32",
        "format": format,
        "ori_shape": dbn_vector_shape,
        "ori_format": ori_format,
    }

    dw_list = {
        "shape": shape_4d_to_fz(filter_shape),
        "dtype": "float32",
        "format": "FRACTAL_Z",
        "ori_shape": filter_shape,
        "ori_format": ori_format,
    }

    filter_sizes = filter_shape
    strides = stride

    data_format = "NCHW"

    if DEBUG_MODE:
        print(
            kernel_name,
            [
                fm_list,
                dedy_list,
                dw_list,
                dbn_vector_list,
                filter_sizes,
                strides,
                padding,
                dilations,
                groups,
                data_format,
            ],
        )

    return {
        "params": [
            fm_list,
            dedy_list,
            dbn_x_list,
            dbn_vector_list, 
            dbn_vector_list,
            dbn_vector_list,
            dbn_vector_list,
            dbn_vector_list,
            dedy_list,
            dw_list,
            filter_sizes,
            strides,
            padding,
            dilations,
            groups,
            data_format,
        ],
        "case_name": kernel_name,
        "expect": expect,
        "format_expect": [],
        "support_expect": True,
    }


def _gen_conv2d_bp_filter_op_case():
    for test_case in fused_dbn_dw_testcase:
        ut_case.add_case(["Ascend910A"], _gen_trans_data_case(*test_case))

_gen_conv2d_bp_filter_op_case()


if __name__ == "__main__":
    ut_case.run("Ascend910A")
    sys.exit(0)