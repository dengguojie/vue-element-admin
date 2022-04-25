import sys
import math

from te import tvm
import te.lang.cce as tbe
from impl.mul import mul_compute

from impl.avg_pool_v2 import _avg_pool_v2_check_rule,avg_pool_v2_compute1


def _pad_compute(padding, input_h, input_w, stride, window, dilations=(1,1), pads=None, ceil_mode=False):
    """
    Calculate the pad value.
    :param padding: str, SAME or VALID
    :param input_h: int, input h
    :param output_w: int, output w
    :param stride: list, stride attr
    :param window: list, window attr
    :param dilations: list, dilations attr
    :return: pad
    """
    He=(window[0] - 1) * dilations[0] + 1
    We=(window[1] - 1) * dilations[1] + 1
    if padding == "SAME":
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = max(0, (output_h - 1) * stride[0] + He - input_h)
        pad_col = max(0, (output_w - 1) * stride[1] + We - input_w)
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    elif padding=='VALID':
        pad = (0, 0, 0, 0)
    else:
        padt, padb, padl, padr = pads
        if ceil_mode:
            Ho = (input_h - He + padt + padb + strideh - 1) // strideh + 1
            wo = (input_w - We + padl + padr + stridew - 1) // stridew + 1
            padb = max(0, (ho - 1) * strideh + He - input_h - padt)
            padr = max(0, (wo - 1) * stridew + We - input_w - padl)
        else:
            ho = (input_h - He + padt + padb) // strideh + 1
            wo = (input_w - We + padl + padr) // stridew + 1
            padb = max(0, (ho - 1) * strideh + He - input_h - padt)
            padr = max(0, (wo - 1) * stridew + We - input_w - padl)
        pad = (padt, padb, padl, padr)
    
    return pad

def avgpool_v2_mul(x, filter, assist_matrix, y, ksize, strides,
             padding="CALCULATED", pads=(0, 0, 0, 0), data_format="NCHW", global_pooling=False, ceil_mode=False, exclusive=True,
             kernel_name="avg_pool", impl_mode="high_performance"):
    # get shape&dtype
    input_shape = x.get("shape")
    input_dtype = x.get("dtype")
    input_dtype = input_dtype.lower()
    output_dtype = y.get("dtype")
    output_dtype = output_dtype.lower()
    input_format = x.get("format")

    _avg_pool_v2_check_rule(input_shape, input_dtype, output_dtype, input_format, ksize, strides,
                            pads, data_format, kernel_name)
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    split_index = x.get("split_index", 0)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "split_index": split_index,
            "L1_fusion_type": l1_fusion_type}
    is_l1fusion = l1_fusion_type in (0, 1)

    if data_format in ("NHWC",):
        input_h = input_shape[1]
        input_w = input_shape[2]
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        stride_h = strides[1]
        stride_w = strides[2]
    else:
        input_h = input_shape[2]
        input_w = input_shape[3]
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        stride_h = strides[2]
        stride_w = strides[3]
    stride = [stride_h, stride_w]

    if global_pooling:
        ksize = list(ksize)
        if data_format in ("NHWC",):
            ksize[1] = input_h
            ksize[2] = input_w
        else:
            ksize[2] = input_h
            ksize[3] = input_w
        padding = 'VALID'
    if list(pads) == [0, 0, 0, 0] and ksize_h == input_h and ksize_w == input_w:
        if padding == "CALCULATED":
            padding = 'VALID'
        if padding == "SAME" and stride_h == input_h and stride_w == input_w:
            padding = 'VALID'

    tensor_in = tvm.placeholder(input_shape, name="tensor_in", dtype=input_dtype, attrs=attr)
    if filter is not None:
        filter_shape = filter.get("shape")
        filter_dtype = filter.get("dtype").lower()
        filter_shape_5d = filter_shape[0], ksize_h, ksize_w, 16, 16
        filter_in = tvm.placeholder(filter_shape_5d, name="filter_in", dtype=filter_dtype, attrs=attr)

        dilations = (1, 1)
        dsl_flag = False

        pad = _pad_compute(padding, input_h, input_w, [stride_h, stride_w], [ksize_h, ksize_w],
                              dilations, pads, ceil_mode)
        res = tbe.te_compute.depthwise_conv2d_compute(
            tensor_in, filter_in, output_dtype.lower(), stride, pad, dilations,
            {"bias_tensor": None, "dsl_flag": dsl_flag, "offset_x": 0}, None, kernel_name)
        tensor_list = [tensor_in, filter_in]
    else:
        res = avg_pool_v2_compute1(tensor_in, y, ksize, strides, padding, data_format, False, kernel_name, impl_mode)
        tensor_list = [tensor_in]

    # schedule
    with tvm.target.cce():
        assist_matrix_input = tvm.placeholder(assist_matrix["shape"], name="assist_matrix", dtype=output_dtype)
        out = mul_compute(res, assist_matrix_input, None)
        sch = tbe.auto_schedule(out)

    tensor_list.extend([assist_matrix_input, out])
    # build
    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": tensor_list,
              "l1_fusion_option": is_l1fusion}
    #tbe.cce_build_code(sch, config)
    tbe.build(sch, config)






