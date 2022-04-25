# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""pooling series tensorflow operators like max_pool"""
# Third-party Packages
import tbetoolkits
from .dav2tf_registry import register_func
from ...utilities import get_global_storage


def _getPadding(data_format, pads, x_shape, w_shape, dy_shape, strides, dilations):
    padt, padb, padl, padr = pads

    if data_format == 'NHWC':
        N, H, W, C = x_shape
        hk, wk, _, Co = w_shape
        _, strideh, stridew, _ = strides
        _, dilationh, dilationw, _ = dilations
    else:
        N, C, H, W = x_shape
        Co, _, hk, wk = w_shape
        _, _, strideh, stridew = strides
        _, _, dilationh, dilationw = dilations
    He = (hk - 1) * dilationh + 1
    We = (wk - 1) * dilationw + 1
    if dy_shape is None:
        if padt != 0 or padb != 0 or padl != 0 or padr != 0:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
            if padt + padb == max(0, (Ho - 1) * strideh + He - H) and \
                    padl + padr == max(0, (Wo - 1) * stridew + We - W):
                padding = 'SAME'
            else:
                padding = 'CALCULATED'
                Ho = (H + padt + padb - He) // strideh + 1
                Wo = (W + padl + padr - We) // stridew + 1
                # raise RuntimeError("not support this padding yet")
        else:  # if padt==0 and padb==0 and padl==0 and padr==0:
            Ho = (H - He) // strideh + 1
            Wo = (W - We) // stridew + 1
            padding = 'VALID'
    else:
        if data_format == 'NHWC':
            N, Ho, Wo, Co = dy_shape
        else:
            N, Co, Ho, Wo = dy_shape
        if Ho == (H + strideh - 1) // strideh \
                and Wo == (W + stridew - 1) // stridew \
                and padt + padb == max(0, (Ho - 1) * strideh + He - H) \
                and padl + padr == max(0, (Wo - 1) * stridew + We - W):
            padding = 'SAME'
        elif Ho == (H - He) // strideh + 1 and \
                Wo == (W - We) // stridew + 1 and \
                padt == 0 and padb == 0 and padl == 0 and padr == 0:
            padding = 'VALID'
        else:
            padding = 'CALCULATED'
            print("not support this padding yet")
            # raise RuntimeError("not support this padding yet")

    return padding


def param_change(context: "tbetoolkits.UniversalTestcaseStructure", name, new_name):
    """Change param"""
    if name in context.other_runtime_params:
        params = context.other_runtime_params
    elif name in context.other_compilation_params:
        params = context.other_compilation_params
    else:
        raise RuntimeError("%s not found in any params" % name)
    params[new_name] = params[name]
    del params[name]

@register_func(["depthwise_conv2d_backprop_filter", ])
def _dp_conv2d_backprop_filter_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "filter_size", "filter_sizes")
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    filter_ori_format = context.output_ori_formats[0]
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    if filter_ori_format == "NCHW":
        params['filter_sizes'] = [params['filter_sizes'][2],params['filter_sizes'][3],params['filter_sizes'][1],params['filter_sizes'][0]]
    elif filter_ori_format == "NHWC":
        params['filter_sizes'] = [params['filter_sizes'][1],params['filter_sizes'][2],params['filter_sizes'][3],params['filter_sizes'][0]]
    elif filter_ori_format == "HWCN":
        params['filter_sizes'] = [params['filter_sizes'][0],params['filter_sizes'][1],params['filter_sizes'][3],params['filter_sizes'][2]]
    return context

@register_func(["conv2d_backprop_filter", ])
def _conv2d_backprop_filter_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "filter_size", "filter_sizes")
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    filter_ori_format = context.output_ori_formats[0]
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    if filter_ori_format == "NCHW":
        params['filter_sizes'] = [params['filter_sizes'][2],params['filter_sizes'][3],params['filter_sizes'][1],params['filter_sizes'][0]]
    elif filter_ori_format == "NHWC":
        params['filter_sizes'] = [params['filter_sizes'][1],params['filter_sizes'][2],params['filter_sizes'][3],params['filter_sizes'][0]]
    return context

@register_func(["conv2d_backprop_input", "depthwise_conv2d_backprop_input"])
def _conv2d_backprop_input_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "input_size", "input_sizes")
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    return context

@register_func(["conv3d_backprop_filter", ])
def _conv3d_backprop_filter_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "filter_size", "filter_sizes")
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] and params['padding'][4] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    context.op_name = "conv3d_backprop_filter_v2"
    return context

@register_func(["conv3d_backprop_input", ])
def _conv3d_backprop_input_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "input_size", "input_sizes")
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] and params['padding'][4] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    context.op_name = "conv3d_backprop_input_v2"
    return context

@register_func(["avg_pool3d", ])
def _avg_pool3d_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] and params['padding'][4] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    return context

@register_func(["conv3d", ])
def _conv3d_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    param_change(context, "pads", "padding")
    params = context.other_runtime_params
    if params['padding'][0] == 0 and params['padding'][1] == 0 and params['padding'][2] == 0 and params['padding'][3] and params['padding'][4] == 0:
        params['padding'] = "VALID"
    else:
        params['padding'] = "SAME"
    # for gpu, need to delete two None for stc_ori_inputs
    context.stc_ori_inputs = context.stc_ori_inputs[:2]
    return context

@register_func(["mat_mul", "batch_matmul_v2", "batch_matmul"])
def _mat_mul_dav2tf(context: "tbetoolkits.UniversalTestcaseStructure"):
    is_gpu = get_global_storage().mode.is_gpu()
    if not is_gpu:
        return context
    # for gpu, need to delete two None for stc_ori_inputs
    context.stc_ori_inputs = context.stc_ori_inputs[:2]
    # change other_runtime_params to adapt gpu
    if context.op_name == "mat_mul":
        param_change(context, "trans_a", "transpose_a")
        param_change(context, "trans_b", "transpose_b")
    if context.op_name == "batch_matmul_v2":
        context.op_name = "batch_mat_mul_v2"
        param_change(context, "trans_a", "adj_x")
        param_change(context, "trans_b", "adj_y")
    if context.op_name == "batch_matmul":
        context.op_name = "batch_mat_mul"
        param_change(context, "trans_a", "adj_x")
        param_change(context, "trans_b", "adj_y")
    return context