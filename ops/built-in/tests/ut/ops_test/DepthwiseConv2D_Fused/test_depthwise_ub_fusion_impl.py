#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from tbe.dsl import auto_schedule
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads, dtype, offset_x):
    bias_shape = bias_shape if bias_shape else []
    kernel_name = 'dp_conv2d_' + '_'.join(map(str, x_shape)) + '_' + '_'.join(map(str, filter_shape)) + '_' + '_'.join(
        map(str, bias_shape)) + '_' + str(stride) + '_' + str(dilation) + "_" + '_'.join(map(
            str, pads)) + '_' + dtype + '_' + str(offset_x)
    return kernel_name


def _shape_to_NC1HWC0(shape, data_format, dtype):
    if data_format.upper() == "NCHW":
        n, c, h, w = shape
    else:  # NCHW
        n, h, w, c = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (n, c1, h, w, c0)


def _shape_to_C1HWNCoC0(shape, data_format, dtype):
    if data_format.upper() == "HWCN":
        h, w, c, n = shape
    else:  # NCHW
        n, c, h, w = shape
    c0 = 16 if dtype.lower() == "float16" else 32
    c1 = (c + c0 - 1) // c0
    return (c1, h, w, n, c0, c0)


def _gen_trans_data_case(param):
    x_shape, filter_shape, bias_shape, y_shape, stride, dilation, pads, data_format, offset_x, dtype, expect_result = param

    data_format = data_format.upper()
    dtype = dtype.lower()
    bias_dtype = "float32" if dtype == "float16" else "int32"
    y_dtype = "float16" if dtype == "float16" else "int32"

    x = {
        "shape": _shape_to_NC1HWC0(x_shape, data_format, dtype),
        "format": "NC1HWC0",
        "ori_shape": x_shape,
        "ori_format": data_format,
        "dtype": dtype
    }
    filter = {
        "shape": _shape_to_C1HWNCoC0(filter_shape, "HWCN", dtype),
        "ori_shape": filter_shape,
        "ori_format": "HWCN",
        "format": "C1HWNCoC0",
        "dtype": dtype
    }
    bias = {
        "shape": _shape_to_NC1HWC0(bias_shape, data_format, bias_dtype),
        "format": "NC1HWC0",
        "ori_shape": bias_shape,
        "ori_format": data_format,
        "dtype": bias_dtype
    } if bias_shape else None
    offset_w = None
    y = {
        "shape": _shape_to_NC1HWC0(y_shape, data_format, y_dtype),
        "format": "NC1HWC0",
        "ori_shape": y_shape,
        "ori_format": data_format,
        "dtype": y_dtype
    }
    strides = [1, stride, stride, 1] if data_format == "NHWC" else [1, 1, stride, stride]
    dilations = [1, dilation, dilation, 1] if data_format == "NHWC" else [1, 1, dilation, dilation]

    kernel_name = _get_kernel_name(x_shape, filter_shape, bias_shape, stride, dilation, pads, dtype, offset_x)
    return {
        "params": [x, filter, bias, offset_w, y, strides, dilations, pads, data_format, offset_x],
        "case_name": kernel_name,
        "expect": expect_result,
        "format_expect": [],
        "support_expect": True
    }


def test_transdata_depthwise_fixpipe_sub_round_rsqrt_square_relu6_fusion(test_arg):
    from impl.depthwise_conv2d import depthwise_compute
    from impl.fix_pipe import fixpipe_compute
    from impl.sub import sub_compute
    from impl.round import round_compute
    from impl.rsqrt import rsqrt_compute
    from impl.square import square_compute
    from impl.relu6 import relu6_compute
    from impl.fixpipe_op.fixpipe_util import create_placeholder
    from tbe import tvm
    import tbe
    import te

    with tbe.common.context.op_context.OpContext(None):
        case=((1, 64, 56, 56), (3, 3, 1, 64), None, (1, 64, 56, 56), 1, 1, (1, 1, 1, 1), "NCHW", 0, "float16", "success")
        res = _gen_trans_data_case(case)
        x = res["params"][0]
        weights = res["params"][1]
        bias = res["params"][2]
        offset_w = res["params"][3]
        outputs = res["params"][4]
        strides = res["params"][5]
        dilations = res["params"][6]
        pads = res["params"][7]
        data_format = res["params"][8]
        offset_x = res["params"][9]
        kernel_name = res["case_name"]
        groups = 1
        filter_fracz = ((weights['ori_shape'][0] * weights['ori_shape'][1] * weights['ori_shape'][3] + 15) // 16,
                        (weights['ori_shape'][2] + 15) // 16, 16, 16)
        weights['format'] = "FRACTAL_Z"
        weights['current_shape'] = filter_fracz
        fmap = tvm.placeholder(x['shape'], dtype=x['dtype'], name='fmap', attrs=x)
        filter_w = tvm.placeholder(filter_fracz, dtype = weights['dtype'], name = 'filter_w', attrs = weights)

        input_y = tvm.placeholder((1,), name="input_y", dtype="float16")
        output_z = None

        out_depthwise = depthwise_compute(fmap, filter_w, bias, offset_w, outputs, strides, dilations, pads,
                                          data_format, offset_x, kernel_name)
        strides = [1, 1, strides[0], strides[1]]
        Ni, Ci, Hi, Wi = x['ori_shape']
        Hk, Wk, Co, _ = weights['ori_shape']
        ho = (Hi + (pads[0] + pads[1]) - Hk) // strides[0] + 1
        wo = (Wi + (pads[2] + pads[3]) - Wk) // strides[1] + 1
        shape_out = out_depthwise.shape
        shape_out_5hd = [shape_out[0], shape_out[1], ho, wo, shape_out[3]]
        output = {
            "shape": shape_out_5hd,
            "format": "NC1HWC0",
            "dtype": "float16"
        }

        x2_dict = {
            "shape": [1, 16, 112, 112],
            "format": "NC1HWC0",
            "dtype": "float16",
            "ori_shape": [1, 16, 112, 112]
        }

        x2 = create_placeholder(x2_dict, "eltwise_src")
        out_fixpipe = fixpipe_compute(out_depthwise, x2, None, None, None, None, None, None, None, None, output, [], ["pre_act"], "SUB")
        out_sub = sub_compute(out_fixpipe, input_y, output_z)
        out_round = round_compute(out_sub, None)
        out_rsqrt = rsqrt_compute(out_round, None)
        out_square = square_compute(out_rsqrt, None)
        out = relu6_compute(out_square, None)

        with te.tvm.target.cce():
            sch = te.utils.cce.auto_schedule(out)


print("adding Conv2D test_transdata_depthwise_fixpipe_sub_round_rsqrt_square_relu6_fusion ut testcases")
ut_case.add_cust_test_func("Ascend310B1", test_func=test_transdata_depthwise_fixpipe_sub_round_rsqrt_square_relu6_fusion)
ut_case.run(['Ascend310B1'])