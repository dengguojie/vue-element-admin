#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for fusion convolution op pattern
"""
# Third-Party Packages
import numpy as np
import tbetoolkits
from ..registry import register_golden


@register_golden(["conv2d"])
def _conv2d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # 5HD to HWCN
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    # x filter to NHWC
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    if groups > 1:
        Cout_ori = (conv_filter.shape[3] + 15) // 16 * 16
        Cout_per_group = Cout_ori // groups
        Cin_ori = (conv_filter.shape[2] + 15) // 16 * 16
        Cin_per_group = Cin_ori // groups
        split_data = tf.split(value=tensor_x,
                              num_or_size_splits=groups,
                              axis=3,
                              name='split1')
        if bias is not None:
            bias = bias.astype(np.float32)
        split_per_group_conv = []
        split_group_index = 0
        for data in split_data:

            conv_filter_per_group = \
                conv_filter[:, :, 0:Cin_per_group,
                split_group_index * Cout_per_group: (split_group_index + 1) * Cout_per_group]

            tf_conv2d_result = tf.nn.conv2d(data, conv_filter_per_group,
                                            strides=(strideh, stridew),
                                            padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                            data_format="NHWC",
                                            use_cudnn_on_gpu=False,
                                            dilations=dilations)
            if bias is not None:
                bias_split_data = \
                    bias[split_group_index * Cout_per_group:(split_group_index + 1) * Cout_per_group]
                tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, bias_split_data)
            split_group_index += 1
            split_per_group_conv.append(tf_conv2d_result)

        output = tf.concat(split_per_group_conv, axis=3, name='concat1')
        init_op = tf.compat.v1.global_variables_initializer()
        feed_dict = {tensor_x: x}
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(output, feed_dict=feed_dict)
    else:
        # Group <= 1
        tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
        tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                        padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                        data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
        feed_dict = {tensor_x: x, tensor_filter: conv_filter}
        if bias is not None:
            bias = bias.astype(np.float32)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
        fusion_mode = None
        if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
            out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).astype(np.float16)
    return output


@register_golden(["depthwise_conv2d"])
def _depthwise_conv2d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")

    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape

    # filter format C,kH,kW,K,C0
    WC, WH, WW, WN, _ = conv_filter.shape
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    if OH == IH and OW == IW:
        pad_mode = "SAME"
    else:
        pad_mode = "VALID"

    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)

    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)

    conv_filter = conv_filter.transpose(1, 2, 0, 3, 4).reshape(WH, WW, WC, WN*C0).astype(np.float32)

    conv_filter = conv_filter[:, :, :, :WN]
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.depthwise_conv2d_native(tensor_x, tensor_filter,
                                                     strides=(1, strideh, stridew, 1),
                                                     padding=pad_mode,
                                                     data_format="NHWC",
                                                     dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)

    # NHWC to NC1HWC0
    ON, OH, OW, OC = out.shape
    OC1 = (OC+15)//C0
    output = out.reshape((ON, OH, OW, OC1, C0)).transpose(0, 3, 1, 2, 4).astype(np.float16)
    return output


@register_golden(["conv2d_leaky_relu"])
def _conv2d_leaky_relu(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.leaky_relu(tf_conv2d_result, 0.1)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "leaky_relu" or fusion_mode == "conv_leaky_relu"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_relu6"])
def _conv2d_relu6(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.relu6(tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "relu6" or fusion_mode == "conv_relu6"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_mul"])
def _conv2d_mul(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.multiply(tf_conv2d_result, tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "mul" or fusion_mode == "conv_mul"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_add"])
def _conv2d_add(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias

    tf_conv2d_result1 = tf.add(tf_conv2d_result, tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result1, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "add" or fusion_mode == "conv_add"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_sigmoid"])
def _conv2d_sigmoid(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.sigmoid(tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "sigmoid" or fusion_mode == "conv_sigmoid"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_softplus"])
def _conv2d_softplus(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.softplus(tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "softplus" or fusion_mode == "conv_softplus"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_add_relu"])
def _conv2d_add_relu(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias

    tf_conv2d_result1 = tf.add(tf_conv2d_result, tf_conv2d_result)
    tf_conv2d_result1 = tf.nn.relu(tf_conv2d_result1)

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result1, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "add_relu" or fusion_mode == "conv_add_relu"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_leaky_relu_add"])
def _conv2d_leaky_relu_add(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias

    tf_conv2d_result = tf.nn.leaky_relu(tf_conv2d_result, 0.1)
    tf_conv2d_result1 = tf.add(tf_conv2d_result, tf_conv2d_result)

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result1, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "leaky_relu_add" or fusion_mode == "conv_leaky_relu_add"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_sigmoid_mul"])
def _conv2d_sigmoid_mul(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape infe
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.sigmoid(tf_conv2d_result)
    tf_conv2d_result = tf.multiply(tf_conv2d_result, tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "sigmoid_mul" or fusion_mode == "conv_sigmoid_mul"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["leaky_relun0_conv2d"])
def _leaky_relun0_conv2d(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<conv_filter_ori.shape>>>>>>>>"
          ">>>>>>>>>>>>>>>>>>>", conv_filter.shape)
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)

    tf_conv2d_result = tf.nn.leaky_relu(tensor_x, 0)
    tf_conv2d_result1 = tf.nn.conv2d(tf_conv2d_result, tensor_filter, strides=(strideh, stridew),
                                     padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                     data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<conv_filter.shape>>>>>>>>>>>>>>>>"
          ">>>>>>>>>>>,", conv_filter.shape)
    if bias is not None:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<bias_add>>>>>>>>>>>>>>>>>>>>>"
              ">>>>>>,bias.shape", bias.shape)
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result1 = tf.nn.bias_add(tf_conv2d_result1, tensor_bias)

        feed_dict[tensor_bias] = bias

    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result1, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "leaky_relun0" or fusion_mode == "leaky_relun0_conv2d"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output


@register_golden(["conv2d_relu"])
def _conv2d_relu(context: "tbetoolkits.UniversalTestcaseStructure"):
    import tensorflow as tf
    x, conv_filter, bias, offset_w = context.input_arrays
    strides = context.other_runtime_params.get("strides")
    pads = context.other_runtime_params.get("pads")
    dilations = context.other_runtime_params.get("dilations")
    groups = context.other_runtime_params.get('groups', 1)
    data_format = context.other_runtime_params.get("data_format", "NCHW")
    offset_x = context.other_runtime_params.get("offset_x", 0)
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError("conv2d testcase golden function supports NC1HWC0 input only!")
    # Collect shape info
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides[h_index], strides[w_index]
    dilationh, dilationw = dilations[h_index], dilations[w_index]
    IN, IC, IH, IW, C0 = x.shape
    WC, WH, WW, WN, _ = conv_filter.shape
    ON = IN
    OC = WN
    OH = (IH + pad_top + pad_bottom - (dilationh * (WH - 1) + 1)) // strideh + 1
    OW = (IW + pad_left + pad_right - (dilationw * (WW - 1) + 1)) // stridew + 1
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    conv_filter = conv_filter.transpose(1, 2, 0, 4, 3).reshape(WH, WW, WC * C0, WN).astype(np.float32)
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    tensor_filter = tf.compat.v1.placeholder(conv_filter.dtype, shape=conv_filter.shape)
    tf_conv2d_result = tf.nn.conv2d(tensor_x, tensor_filter, strides=(strideh, stridew),
                                    padding=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    data_format="NHWC", use_cudnn_on_gpu=False, dilations=dilations)
    feed_dict = {tensor_x: x, tensor_filter: conv_filter}
    if bias is not None:
        bias = bias.astype(np.float32)
        tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
        tf_conv2d_result = tf.nn.bias_add(tf_conv2d_result, tensor_bias)
        feed_dict[tensor_bias] = bias
    tf_conv2d_result = tf.nn.relu(tf_conv2d_result)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(tf_conv2d_result, feed_dict=feed_dict)
    fusion_mode = None
    if fusion_mode is not None and (fusion_mode == "relu" or fusion_mode == "conv_relu"):
        out = np.maximum(out, 0)
    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4).copy().astype(np.float16)
    return output
