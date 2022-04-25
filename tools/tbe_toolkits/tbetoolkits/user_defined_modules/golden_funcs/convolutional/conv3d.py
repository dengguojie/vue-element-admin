#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for convolution pattern
"""
# Third-Party Packages
import numpy as np
import tbetoolkits
from .convolutional_utils import _ceil
from .convolutional_utils import _calculate_group
from .convolutional_utils import _align

from ..registry import register_golden


@register_golden(["conv3d"])
def _conv3d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # noinspection PyUnresolvedReferences
    import torch
    # 5HD input only
    if len(x.shape) != 6:
        raise RuntimeError("conv3d testcase golden function supports NDC1HWC0 input only!")
    # Collect shape info

    block_size = 16
    fmap_ori_shape, weight_ori_shape, _, _ = ori_shapes
    fmap_ori_format, weight_ori_format, _, _ = ori_formats
    cout_ori = weight_ori_shape[weight_ori_format.index("N")]
    cin_ori = fmap_ori_shape[fmap_ori_format.index("C")]

    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = pads
    strided, strideh, stridew = strides[d_index], strides[h_index], strides[w_index]
    dilationd, dilationh, dilationw = dilations[d_index], dilations[h_index], dilations[w_index]

    IN, ID, IC, IH, IW, C0 = x.shape
    real_g, WD, cin1_g, WH, WW, cout_g, _ = conv_filter.shape

    ON = IN
    OC = _align(cout_ori, block_size)
    OD = (ID + pad_head + pad_tail - (dilationd * (WD - 1) + 1)) // strided + 1
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    if groups == 1:
        print("######################### Using tensorflow ############################")
        WN = cout_g
        # x filter to NDHWC
        x = x.transpose(0, 1, 3, 4, 2, 5).reshape(IN, ID, IH, IW, IC * C0).astype(np.float32)
        # 5HD to DHWCN
        conv_filter = conv_filter.transpose(1, 3, 4, 0, 2, 6, 5).reshape(WD, WH, WW, real_g * cin1_g * C0, WN).astype(
            np.float32)
        if list(pads) == [0, 0, 0, 0, 0, 0]:
            padding = "VALID"
        else:
            padding = "SAME"
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
        tf_conv3d_result = tf.compat.v1.nn.conv3d(tensor_x, tensor_filter, strides=[1, strided, strideh, stridew, 1],
                                                  padding=padding, data_format="NDHWC",
                                                  dilations=[1, dilationd, dilationh, dilationw, 1])
        feed_dict = {tensor_x: x, tensor_filter: conv_filter}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_conv3d_result, feed_dict=feed_dict)
    else:
        # Recover filter to NCDHW
        print("######################### Using Pytorch ############################")
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        mag_factor = group_dict['mag_factor']

        filter_c = cin_ori // groups
        weight_ori_ncdhw = (cout_ori, filter_c, WD, WH, WW)
        weight = np.zeros(weight_ori_ncdhw)

        # Recover Filter Data
        for g in range(groups):
            for ci in range(filter_c):
                for co in range(cout_ori // groups):
                    e = g % mag_factor
                    dst_cin = e * filter_c + ci
                    dst_cout = e * (cout_ori // groups) + co
                    src_cout = g * (cout_ori // groups) + co
                    weight[src_cout, ci, :, :, :] = conv_filter[g // mag_factor, :, dst_cin // block_size, :, :,
                                                    dst_cout, dst_cin % block_size]

        # Recover Fmap to NCDHW
        fmap_ori = x.transpose(0, 2, 5, 1, 3, 4).reshape(IN, IC * C0, ID, IH, IW).astype(np.float32).copy()
        # fmap_ncdhw = np.zeros((IN, cin_ori, ID, IH, IW))
        fmap_ncdhw = fmap_ori[:, :cin_ori, :, :, :]
        # Padding compensation
        d_pad_diff = pad_tail - pad_head
        h_pad_diff = pad_bottom - pad_top
        w_pad_diff = pad_right - pad_left
        temp = np.zeros((IN, cin_ori, ID + d_pad_diff, IH + h_pad_diff, IW + w_pad_diff))
        temp[:, :, :ID, :IH, :IW] = fmap_ncdhw
        fmap_torch = temp
        pad_torch = [pad_head, pad_top, pad_left]

        fmap_torch = torch.from_numpy(fmap_torch).type(torch.float32)
        weight_torch = torch.from_numpy(weight).type(torch.float32)
        stridedhw = [strided, strideh, stridew]
        dilationdhw = [dilationd, dilationh, dilationw]
        # out Shape in Torch is NCDHW
        out = torch.nn.functional.conv3d(fmap_torch, weight_torch, stride=stridedhw, padding=pad_torch,
                                         groups=groups, dilation=dilationdhw)
        out = out.numpy()
        # Padding OutPut C channel
        zero_padding_in_Cout = _ceil(cout_ori, block_size) * block_size
        _, _, out_d, out_h, out_w = out.shape
        out_pad = np.zeros((IN, zero_padding_in_Cout, out_d, out_h, out_w))
        out_pad[:, :cout_ori, :, :, :] = out

        out = out_pad.transpose((0, 2, 3, 4, 1))  # NCDHW Change To NDHWC
    if output_dtype[0] == 'float16':
        # operation for overflow
        out = np.maximum(out, -65504)
        out = np.minimum(out, 65504)
        out = out.astype(np.float16)
        if bias is not None:
            tensor_out = tf.compat.v1.placeholder(out.dtype, shape=out.shape)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            out_bias = tf.compat.v1.nn.bias_add(tensor_out, tensor_bias)
            feed_dict = {tensor_out: out, tensor_bias: bias}
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                # Generate output tf data
                out = sess.run(out_bias, feed_dict=feed_dict)
            # operation for overflow
            out = np.maximum(out, -65504)
            out = np.minimum(out, 65504)
        fusion_mode = None
        if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
            out = np.maximum(out, 0)

        # NDHWC to NDC1HWC0
        output = out.reshape((ON, OD, OH, OW, OC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float16)
    elif output_dtype[0] == 'float32':
        out = out.astype(np.float32)
        if bias is not None:
            tensor_out = tf.compat.v1.placeholder(out.dtype, shape=out.shape)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            out_bias = tf.compat.v1.nn.bias_add(tensor_out, tensor_bias)
            feed_dict = {tensor_out: out, tensor_bias: bias}
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                # Generate output tf data
                out = sess.run(out_bias, feed_dict=feed_dict)
        fusion_mode = None
        if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
            out = np.maximum(out, 0)

        # NDHWC to NDC1HWC0
        output = out.reshape((ON, OD, OH, OW, OC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float32)
    else:
        raise TypeError("Unsupported type for conv3d: %s" % output_dtype[0])
    return output


# noinspection PyUnusedLocal
@register_golden(["conv3d_transpose"])
def _conv3d_transpose(context: "tbetoolkits.UniversalTestcaseStructure"):
    _, out_backprop, conv_filter, bias, offset_w = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    output_padding = context.other_runtime_params.get("output_padding", (0, 0, 0, 0, 0))
    offset_x = context.other_runtime_params.get("offset_x", 0)
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # noinspection PyUnresolvedReferences
    import torch
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    # 6HD input only
    # if len(out_backprop.shape) != 6:
    #    raise RuntimeError("conv3d testcase golden function supports NDC1HWC0 input only!")
    # Collect shape info
    block_size = 16
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")
    strided, strideh, stridew = strides[d_index], strides[h_index], strides[w_index]
    dilationd, dilationh, dilationw = dilations[d_index], dilations[h_index], dilations[w_index]
    IN, ID, IH, IW, IC_ori = (input_size[n_index], input_size[d_index], input_size[h_index], input_size[w_index],
                              input_size[c_index])
    IC = (IC_ori + 15) // 16 * 16
    YN, YD, YC, YH, YW, C0 = out_backprop.shape
    real_g, w_d, cin1_g, w_h, w_w, cout_g, c0 = conv_filter.shape
    dy_ori_shape, filter_ori_shape, bias_shape, offset_w_shape = ori_shapes
    dy_ori_format, filter_ori_format, bias_format, offset_w_format = ori_formats
    dy_ori_c = dy_ori_shape[dy_ori_format.index("C")]
    filter_ori_c = filter_ori_shape[filter_ori_format.index("C")]
    # backprop to NDHWC
    filter_data = np.zeros((w_d, w_h, w_w, filter_ori_c, dy_ori_c), dtype=np.float16)
    out_backprop = out_backprop.transpose(0, 1, 3, 4, 2, 5).reshape(YN, YD, YH, YW, YC * C0).astype(np.float32)
    # backprop to ori_shape
    out_backprop_orishape = out_backprop[:, :, :, :, :dy_ori_c]
    group_dict = _calculate_group(IC_ori, dy_ori_c, groups)
    mag_factor = group_dict["mag_factor"]
    print("###########################", "\n filter_data", filter_data.shape, "\n conv_filter", conv_filter.shape,
          "\n mag_factor", mag_factor)
    # 7HD to DHWCN
    for g in range(groups):
        for ci in range(filter_ori_c):
            for co in range(dy_ori_c // groups):
                e = g % mag_factor
                dst_cin = e * filter_ori_c + ci
                dst_cout = e * (dy_ori_c // groups) + co
                src_cout = g * (dy_ori_c // groups) + co
                # print("ci, src_cout,g // mag_factor, dst_cin // block_size, dst_cout, dst_cinblock_size",
                #       ci, src_cout,g // mag_factor, dst_cin // block_size, dst_cout, dst_cin%block_size)
                filter_data[:, :, :, ci, src_cout] = conv_filter[g // mag_factor, :, dst_cin // block_size, :, :,
                                                     dst_cout, dst_cin % block_size]
    filter_data = filter_data[:, :, :, :filter_ori_c, :]
    filter_data.tofile("filter.bin")
    print("1111111111111111111111111111111111111111111111111111finish fraz")
    # =================== pytorch ===================
    pytorch_input_data = out_backprop_orishape  # NDHWC  -- > NCDHW
    pytorch_filter_data = filter_data  # DHWCN  -- > CNDHW
    pytorch_input_data = torch.from_numpy(pytorch_input_data.transpose(0, 4, 1, 2, 3)).type(torch.float32)
    pytorch_filter_data = torch.from_numpy(pytorch_filter_data.transpose(4, 3, 0, 1, 2)).type(torch.float32)

    strides_dhw = [strided, strideh, stridew]
    dilations_dhw = [dilationd, dilationh, dilationw]
    input_dhw = [YD, YH, YW]
    out_dhw = [ID, IH, IW]
    filter_dhw = [w_d, w_h, w_w]
    filter_dilated_dhw = list(((k - 1) * d + 1) for k, d in zip(filter_dhw, dilations_dhw))
    # pads = _get_pads_3d(out_dhw, filter_dilated_dhw, strides_dhw, self.padding)
    padding = [pads[0], pads[2], pads[4]]
    # pad_before = list(k - 1 - p for k, p in zip(filter_dilated_dhw, [pads[0], pads[2], pads[4]]))
    # output_padding = list((i + 2 * p - k) % s for i, p, k, s in
    #                       zip(out_dhw, pad_before, filter_dhw, strides_dhw))
    # out = (input - 1) * stride - 2 * padding + kernel_size + out_padding
    output_padding = list(o - (i - 1) * s + 2 * p - k for o, i, s, p, k in
                          zip(out_dhw,
                              input_dhw,
                              strides_dhw,
                              padding,
                              filter_dilated_dhw))
    output_padding = [a if a > b else b for a, b in zip([0] * 3, output_padding)]
    print("(pytorch_input_data", pytorch_input_data.shape, "pytorch_filter_data", pytorch_filter_data.shape, "groups",
          groups, "stride", strides_dhw, "padding", padding,
          "output_padding", output_padding, "dilation==", dilations_dhw)
    pytorch_out = torch.nn.functional.conv_transpose3d(pytorch_input_data, pytorch_filter_data, groups=groups,
                                                       stride=strides_dhw, padding=padding,
                                                       output_padding=output_padding, dilation=dilations_dhw)  # NCDHW
    zero_shape = (IN, ID, IH, IW, IC)
    zero_out = np.zeros(zero_shape, dtype=np.float32)
    zero_out[:, :, :, :, :IC_ori] = pytorch_out.numpy()[:, :, :ID, :IH, :IW].transpose(0, 2, 3, 4, 1)
    if output_dtype[0] == 'float16':
        # operation for overflow
        zero_out = np.maximum(zero_out, -65504)
        zero_out = np.minimum(zero_out, 65504)
        zero_out = zero_out.astype(np.float16)
        if bias is not None:
            tensor_out = tf.compat.v1.placeholder(zero_out.dtype, shape=zero_out.shape)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            out_bias = tf.compat.v1.nn.bias_add(tensor_out, tensor_bias)
            feed_dict = {tensor_out: zero_out, tensor_bias: bias}
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                # Generate output tf data
                zero_out = sess.run(out_bias, feed_dict=feed_dict)

        # NDHWC to NDC1HWC0
        output = zero_out.reshape((IN, ID, IH, IW, IC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float16)
    elif output_dtype[0] == 'float32':
        zero_out = zero_out.astype(np.float32)
        if bias is not None:
            tensor_out = tf.compat.v1.placeholder(zero_out.dtype, shape=zero_out.shape)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            out_bias = tf.compat.v1.nn.bias_add(tensor_out, tensor_bias)
            feed_dict = {tensor_out: zero_out, tensor_bias: bias}
            init_op = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:
                sess.run(init_op)
                # Generate output tf data
                zero_out = sess.run(out_bias, feed_dict=feed_dict)
        # NDHWC to NDC1HWC0
        output = zero_out.reshape((IN, ID, IH, IW, IC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float32)
    else:
        raise TypeError("Unsupported type for conv3d: %s" % output_dtype[0])
    return output


@register_golden(["conv3d_backprop_input"])
def _conv3d_backprop_input(context: "tbetoolkits.UniversalTestcaseStructure"):
    conv_filter, out_backprop = context.input_arrays
    input_size = context.other_runtime_params.get("input_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    ori_shapes = context.stc_ori_inputs
    ori_formats = context.stc_input_ori_formats
    output_dtype = context.output_dtypes
    # noinspection PyUnresolvedReferences
    import torch
    # 6HD input only
    # if len(out_backprop.shape) != 6:
    #    raise RuntimeError("conv3d testcase golden function supports NDC1HWC0 input only!")
    # Collect shape info
    block_size = 16
    n_index = data_format.index("N")
    d_index = data_format.index("D")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    c_index = data_format.index("C")
    strided, strideh, stridew = strides[d_index], strides[h_index], strides[w_index]
    dilationd, dilationh, dilationw = dilations[d_index], dilations[h_index], dilations[w_index]
    IN, ID, IH, IW, IC_ori = (input_size[n_index], input_size[d_index], input_size[h_index], input_size[w_index],
                              input_size[c_index])
    IC = (IC_ori + 15) // 16 * 16
    YN, YD, YC, YH, YW, C0 = out_backprop.shape
    real_g, w_d, cin1_g, w_h, w_w, cout_g, c0 = conv_filter.shape
    filter_ori_shape, dy_ori_shape = ori_shapes
    filter_ori_format, dy_ori_format = ori_formats
    dy_ori_c = dy_ori_shape[dy_ori_format.index("C")]
    filter_ori_c = filter_ori_shape[filter_ori_format.index("C")]
    # backprop to NDHWC
    filter_data = np.zeros((w_d, w_h, w_w, filter_ori_c, dy_ori_c), dtype=np.float16)
    out_backprop = out_backprop.transpose(0, 1, 3, 4, 2, 5).reshape(YN, YD, YH, YW, YC * C0).astype(np.float32)
    # backprop to ori_shape
    out_backprop_orishape = out_backprop[:, :, :, :, :dy_ori_c]
    group_dict = _calculate_group(IC_ori, dy_ori_c, groups)
    mag_factor = group_dict["mag_factor"]
    print("###########################", "\n filter_data", filter_data.shape, "\n conv_filter", conv_filter.shape,
          "\n mag_factor", mag_factor)
    # 7HD to DHWCN
    for g in range(groups):
        for ci in range(filter_ori_c):
            for co in range(dy_ori_c // groups):
                e = g % mag_factor
                dst_cin = e * filter_ori_c + ci
                dst_cout = e * (dy_ori_c // groups) + co
                src_cout = g * (dy_ori_c // groups) + co
                filter_data[:, :, :, ci, src_cout] = conv_filter[g // mag_factor, :, dst_cin // block_size, :, :,
                                                     dst_cout, dst_cin % block_size]
    filter_data = filter_data[:, :, :, :filter_ori_c, :]
    filter_data.tofile("filter.bin")
    print("1111111111111111111111111111111111111111111111111111finish fraz")
    # =================== pytorch ===================
    pytorch_input_data = out_backprop_orishape  # NDHWC  -- > NCDHW
    pytorch_filter_data = filter_data  # DHWCN  -- > CNDHW
    pytorch_input_data = torch.from_numpy(pytorch_input_data.transpose(0, 4, 1, 2, 3)).type(torch.float32)
    pytorch_filter_data = torch.from_numpy(pytorch_filter_data.transpose(4, 3, 0, 1, 2)).type(torch.float32)

    strides_dhw = [strided, strideh, stridew]
    dilations_dhw = [dilationd, dilationh, dilationw]
    input_dhw = [YD, YH, YW]
    out_dhw = [ID, IH, IW]
    filter_dhw = [w_d, w_h, w_w]
    filter_dilated_dhw = list(((k - 1) * d + 1) for k, d in zip(filter_dhw, dilations_dhw))
    # pads = _get_pads_3d(out_dhw, filter_dilated_dhw, strides_dhw, self.padding)
    padding = [pads[0], pads[2], pads[4]]
    # pad_before = list(k - 1 - p for k, p in zip(filter_dilated_dhw, [pads[0], pads[2], pads[4]]))
    # output_padding = list((i + 2 * p - k) % s for i, p, k, s in
    #                      zip(out_dhw, pad_before, filter_dhw, strides_dhw))
    # out = (input - 1) * stride - 2 * padding + kernel_size + out_padding
    output_padding = list(o - (i - 1) * s + 2 * p - k for o, i, s, p, k in
                          zip(out_dhw,
                              input_dhw,
                              strides_dhw,
                              padding,
                              filter_dilated_dhw))
    output_padding = [a if a > b else b for a, b in zip([0] * 3, output_padding)]
    print("(pytorch_input_data", pytorch_input_data.shape, "pytorch_filter_data", pytorch_filter_data.shape, "groups",
          groups, "stride", strides_dhw, "padding", padding,
          "output_padding", output_padding, "dilation==", dilations_dhw)
    pytorch_out = torch.nn.functional.conv_transpose3d(pytorch_input_data, pytorch_filter_data, groups=groups,
                                                       stride=strides_dhw, padding=padding,
                                                       output_padding=output_padding, dilation=dilations_dhw)  # NCDHW

    zero_shape = (IN, ID, IH, IW, IC)
    zero_out = np.zeros(zero_shape, dtype=np.float32)
    zero_out[:, :, :, :, :IC_ori] = pytorch_out.numpy()[:, :, :ID, :IH, :IW].transpose(0, 2, 3, 4, 1)
    if output_dtype[0] == 'float16':
        # operation for overflow
        zero_out = np.maximum(zero_out, -65504)
        zero_out = np.minimum(zero_out, 65504)
        # NDHWC to NDC1HWC0
        output = zero_out.reshape((IN, ID, IH, IW, IC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float16)
    elif output_dtype[0] == 'float32':
        # NDHWC to NDC1HWC0
        output = zero_out.reshape((IN, ID, IH, IW, IC // C0, C0)).transpose(0, 1, 4, 2, 3, 5).copy().astype(np.float32)
    else:
        raise TypeError("Unsupported type for conv3d: %s" % output_dtype[0])
    return output


@register_golden(["conv3d_backprop_filter"])
def _conv3d_backprop_filter(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    x, out_backprop = context.input_arrays
    filter_size = context.other_runtime_params.get("filter_size")
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NDHWC")
    filter_format = context.output_ori_formats[0]
    # noinspection PyUnresolvedReferences
    import torch
    # noinspection PyUnresolvedReferences
    from torch.autograd import Variable
    if groups == 1:
        print("===============================")
        print(x.shape, out_backprop.shape, filter_size, strides, pads, dilations, groups, data_format)
        # 5HD input only
        if len(x.shape) != 6:
            raise RuntimeError("conv3d testcase golden function supports NC1HWC0 input only!")
        # Collect shape info
        d_index = data_format.index("D")
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        strided, strideh, stridew = strides[d_index], strides[h_index], strides[w_index]
        # dilationh, dilationw = dilations[h_index], dilations[w_index]
        IN, ID, IC, IH, IW, C0 = x.shape
        YN, YD, YC, YH, YW, C0 = out_backprop.shape
        if filter_format == "DHWCN":
            kd, kh, kw, C, Co = filter_size
        elif filter_format == "NCDHW":
            Co, C, kd, kh, kw = filter_size
        elif filter_format == "NDHWC":
            Co, kd, kh, kw, C = filter_size
        if list(pads) == [0, 0, 0, 0, 0, 0]:
            padding = "VALID"
        else:
            padding = "SAME"
        # x filter to NDHWC
        x = x.transpose(0, 1, 3, 4, 2, 5).reshape(IN, ID, IH, IW, IC * C0).astype(np.float32)
        out_backprop = out_backprop.transpose(0, 1, 3, 4, 2, 5).reshape(YN, YD, YH, YW, YC * C0).astype(np.float32)
        Co1 = (Co + C0 - 1) // C0
        w_shape = (kd, kh, kw, IC * C0, Co1 * C0)  # DHWCN

        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        tensor_dy = tf.compat.v1.placeholder(out_backprop.dtype, shape=out_backprop.shape)
        tf_dw_result = tf.compat.v1.nn.conv3d_backprop_filter(tensor_x, w_shape, tensor_dy,
                                                              strides=[1, strided, strideh, stridew, 1],
                                                              padding=padding,
                                                              data_format="NDHWC", dilations=dilations)
        feed_dict = {tensor_x: x, tensor_dy: out_backprop}
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_dw_result, feed_dict=feed_dict)
        # DHWCN to C1HWNC0
        output = out.reshape(kd, kh, kw, IC, C0, YC * C0).transpose(0, 3, 1, 2, 5, 4).copy().astype(
            np.float32)
        return output
    else:
        fmap_n, fmap_d, fmap_c1, fmap_h, fmap_w, fmap_c0 = x.shape
        y_n, y_d, y_c1, y_h, y_w, y_c0 = out_backprop.shape
        if filter_format == "DHWCN":
            w_d, w_h, w_w, w_c, w_n = filter_size
        elif filter_format == "NCDHW":
            w_n, w_c, w_d, w_h, w_w = filter_size
        elif filter_format == "NDHWC":
            w_n, w_d, w_h, w_w, w_c = filter_size
        fmap_c = w_c * groups
        y_c = w_n
        d_index = data_format.index("D")
        h_index = data_format.index("H")
        w_index = data_format.index("W")
        stride_d, stride_h, stride_w = strides[d_index], strides[h_index], strides[w_index]
        dilation_d, dilation_h, dilation_w = dilations[d_index], dilations[h_index], dilations[w_index]
        (pad_front, pad_back, pad_up, pad_down, pad_left, pad_right) = pads
        # sovle asymmetric pad
        zero_pad = list(head - behind for head, behind in zip(pads[1::2], pads[::2]))
        zero_pad_d, zero_pad_h, zero_pad_w = zero_pad

        # torch format is NCDHW
        input_data = x.transpose(0, 2, 5, 1, 3, 4).reshape(fmap_n, fmap_c1 * fmap_c0, fmap_d, fmap_h, fmap_w).astype(
            np.float32)
        input_data1 = np.zeros((fmap_n, fmap_c, fmap_d, fmap_h, fmap_w)).astype(np.float32)
        input_data1[:, :, :, :, :] = input_data[:, :fmap_c, :, :, :]
        input_pad = np.pad(input_data1, ((0, 0), (0, 0), (0, zero_pad_d), (0, zero_pad_h), (0, zero_pad_w)),
                           constant_values=(0, 0))
        weight_data = np.random.uniform(-1, 1, size=(w_n, w_c, w_d, w_h, w_w)).astype(np.float16)

        _input = Variable(torch.from_numpy(input_pad).type(torch.float32), requires_grad=True)
        weight = Variable(torch.from_numpy(weight_data).type(torch.float32), requires_grad=True)

        out = torch.nn.functional.conv3d(_input, weight,
                                         stride=(stride_d, stride_h, stride_w),
                                         padding=(pad_front, pad_up, pad_left),
                                         dilation=(dilation_d, dilation_h, dilation_w),
                                         groups=groups,
                                         bias=None)

        # grad_data = _uniform_data((y_n, y_c, y_d, y_h, y_w), -1, 1, 'float').astype(np.float32)
        grad_data = out_backprop.transpose(0, 2, 5, 1, 3, 4).reshape(y_n, y_c1 * y_c0, y_d, y_h, y_w).astype(np.float32)
        grad_data1 = np.zeros((y_n, y_c, y_d, y_h, y_w)).astype(np.float32)
        grad_data1[:, :, :, :, :] = grad_data[:, :y_c, :, :, :]
        gradients = torch.from_numpy(grad_data1)
        out.backward(gradients, retain_graph=True)

        # trans weight data to FRAC_Z
        weight_np = weight.grad.detach().numpy()
        group_dict = _calculate_group(fmap_c, w_n, groups)
        real_g = group_dict["real_g"]
        cin1_g = group_dict["cin1_g"]
        cout_g = group_dict["cout_g"]
        mag_factor = group_dict["mag_factor"]

        weight_group_data = np.ones((real_g, w_d, cin1_g, w_h, w_w, cout_g, 16)).astype(np.float32) * 999999
        for g in range(groups):
            for ci in range(w_c):
                for co in range(w_n // groups):
                    try:
                        e = g % mag_factor
                        dst_cin = e * w_c + ci
                        dst_cout = e * (w_n // groups) + co
                        src_cout = g * (w_n // groups) + co
                        weight_group_data[g // mag_factor, :, dst_cin // 16, :, :, dst_cout, dst_cin % 16] = \
                            weight_np[src_cout, ci, :, :, :]
                    except:
                        # e = g % mag_factor
                        # dst_cin = e * w_c + ci
                        # dst_cout = e * (w_n // groups) + co
                        # src_cout = g * (w_n // groups) + co
                        # print("================================ Error Detected =====================================")
                        # print("weight_group shape:", weight_group.shape)
                        # print("Weight Shape : ", weight.shape)
                        # print("C0:", co)
                        # print("e : ", e)
                        # print("dst_cin :", dst_cin)
                        # print("dst_cout : ", dst_cout)
                        # print("src_cout and Ci" , src_cout, "",ci)
                        # print("mag_factor : ", mag_factor)
                        raise

        return weight_group_data
