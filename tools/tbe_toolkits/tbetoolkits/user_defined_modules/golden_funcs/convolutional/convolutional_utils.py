#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for convolution pattern
"""
# Third-Party Packages
import numpy as np


def _ceil(a, b):
    return (a + b - 1) // b


def _align(a, b):
    return _ceil(a, b) * b


def _lcm(param1, param2):
    """
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, (param1 % param2)

    return temp // param2


def _calculate_group(cin, cout, groups):
    block_size = 16

    mag_factor0 = _lcm(cin // groups, block_size) // (cin // groups)
    mag_factor1 = _lcm(cout // groups, block_size) // (cout // groups)
    mag_factor = min(_lcm(mag_factor0, mag_factor1), groups)

    cin_g = _align(mag_factor * (cin // groups), block_size)
    cout_g = _align(mag_factor * (cout // groups), block_size)

    group_dict = {"real_g": _ceil(groups, mag_factor),
                  "mag_factor": mag_factor,
                  "cin_g": cin_g,
                  "cin1_g": cin_g // block_size,
                  "cout_g": cout_g,
                  "cout1_g": cout_g // block_size,
                  "groups": groups,
                  "cin_ori": cin // groups,
                  "cout_ori": cout // groups}
    return group_dict


def due_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def _getPadList(padding, x_shape, w_shape, dy_shape, strides, dilations=(1, 1, 1, 1), pads=None, ceil_mode=False):
    N, C, H, W = x_shape
    Co, _, hk, wk = w_shape
    strideh, stridew = strides
    _, _, dilationh, dilationw = dilations
    He = (hk - 1) * dilationh + 1
    We = (wk - 1) * dilationw + 1
    if padding == 'VALID':
        pad = [0, 0, 0, 0]
        if dy_shape is None:
            Ho = (H - He) // strideh + 1
            Wo = (W - We) // stridew + 1
            # dy_shape = [N, Co, Ho, Wo]
        else:
            N, Co, Ho, Wo = dy_shape
    elif padding == 'SAME':
        if dy_shape is None:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
            # dy_shape = [N, Co, Ho, Wo]
        else:
            N, Co, Ho, Wo = dy_shape
        padh = max(0, (Ho - 1) * strideh + He - H)
        padw = max(0, (Wo - 1) * stridew + We - W)
        pad = [padh // 2, padh // 2 + padh % 2, padw // 2, padw // 2 + padw % 2]
    elif padding == "CALCULATED":
        if pads is None:
            raise RuntimeError("pads cannot be None when padding=CALCULATED")
        pad = pads
        if dy_shape is None:
            padt, padb, padl, padr = pads
            if ceil_mode:
                Ho = (H - He + padt + padb + strideh - 1) // strideh + 1
                Wo = (W - We + padl + padr + stridew - 1) // stridew + 1
                # padb = max(0, (ho - 1) * strideh + He - H - padt)
                # padr = max(0, (wo - 1) * stridew + We - W - padl)
            else:
                Ho = (H - He + padt + padb) // strideh + 1
                Wo = (W - We + padl + padr) // stridew + 1
                # padb = max(0, (ho - 1) * strideh + He - H - padt)
                # padr = max(0, (wo - 1) * stridew + We - W - padl)
            # pad = (padt, padb, padl, padr)
        else:
            N, Co, Ho, Wo = dy_shape
    else:
        raise RuntimeError("not support this mode:{} yet".format(padding))

    return pad, [N, Co, Ho, Wo]


def _getPads(padding, x_shape, w_shape, dy_shape, strides, dilations):
    _, H, W, _ = x_shape
    kh, kw, _, _ = w_shape
    strideh, stridew = strides
    dilationh, dilationw = dilations
    He = (kh - 1) * dilationh + 1
    We = (kw - 1) * dilationw + 1
    if padding == 'VALID':
        pads = [0, 0, 0, 0]
    elif padding == 'SAME':
        if dy_shape is None:
            Ho = (H + strideh - 1) // strideh
            Wo = (W + stridew - 1) // stridew
        else:
            _, Ho, Wo, _ = dy_shape
        padh = max(0, (Ho - 1) * strideh + He - H)
        padw = max(0, (Wo - 1) * stridew + We - W)
        pads = [padh // 2, padh - padh // 2, padw // 2, padw - padw // 2]
    else:
        raise RuntimeError('not support this padding yet')

    return pads


def _getPadding(pads, x_shape, w_shape, dy_shape, strides, dilations):
    padt, padb, padl, padr = pads

    N, H, W, C = x_shape
    hk, wk, _, Co = w_shape
    strideh, stridew = strides
    dilationh, dilationw = dilations
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
        N, Ho, Wo, Co = dy_shape
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
            # raise RuntimeError("not support this padding yet")

    return padding, [N, Co, Ho, Wo]


def _native_fun(input_shape, input_f):
    if len(input_shape) == 4:
        Hf, Wf, C, N = input_shape
        if N != 1:
            raise RuntimeError("N != 1.")
    else:
        raise RuntimeError("depthwise_weight_4d_2_6d does not support %dD shape."
                           % (len(input_shape)))

    C0 = 16
    C1 = (C + C0 - 1) // C0
    _filter = NCHW_to_NC1C0HW(input_f.reshape([Hf, Wf, C, 1]).transpose(3, 2, 0, 1)).transpose(1, 3, 4, 0, 2)
    filter_6d = np.zeros((C1, Hf, Wf, 1, 16, C0), dtype=np.float32)
    for d4 in range(C0):
        for d5 in range(C0):
            if d4 == d5:
                filter_6d[:, :, :, 0, d4, d5] = _filter[:, :, :, 0, d5]
    out = filter_6d.transpose(3, 1, 2, 0, 4, 5).reshape(Hf * Wf, C1, 16, 16)
    return out.astype(np.float32)


# generate depthwise conv2d backprop filter input
def _gen_depthwise_conv2d_backprop_filter_data(x, out_backprop, filter_size, strides, pads):
    print("Info: writing input for depthwise_conv2d_backprop_filter...\n")
    N, C1, H, W, C0 = x.shape
    x = x.transpose((0, 2, 3, 1, 4)).reshape((N, H, W, C1 * C0))
    N, C1, H, W, C0 = out_backprop.shape
    out_backprop = out_backprop.transpose((0, 2, 3, 1, 4)).reshape((N, H, W, C1 * C0))
    ori_dfilter_tensor = _depthwise_conv2d_native_backprop_filter(x.astype(np.float32), filter_size,
                                                                  out_backprop.astype(np.float32), strides, pads=pads,
                                                                  dilations=[1, 1, 1, 1])
    h, w, c, n = filter_size
    c0 = 16
    c1 = (c + 15) // 16
    ori_dfilter_tensor = ori_dfilter_tensor.reshape((h, w, c1, c0, n)).transpose((2, 0, 1, 4, 3))
    dfilter_tensor = np.zeros((c1, h, w, n, c0, c0), np.float32)
    for i in range(c0):
        dfilter_tensor[:, :, :, :, i, i] = ori_dfilter_tensor[:, :, :, :, i]
    dfilter_tensor = dfilter_tensor.transpose(3, 1, 2, 0, 4, 5).reshape(n * h * w, c1, c0, c0)
    print("Info: writing output for depthwise_conv2d_backprop_filter done!!!\n")
    return dfilter_tensor


# noinspection PyUnusedLocal
def _depthwise_conv2d_native_backprop_filter(_input,
                                             filter_sizes,
                                             out_backprop,
                                             strides,
                                             pads,
                                             data_format='NHWC',
                                             dilations=(1, 1, 1, 1),
                                             name=None):
    batch = _input.shape[0]
    input_height = _input.shape[1]
    input_width = _input.shape[2]
    in_channels = _input.shape[3]
    filter_height = filter_sizes[0]
    filter_width = filter_sizes[1]
    out_height = out_backprop.shape[1]
    out_width = out_backprop.shape[2]
    out_channels = out_backprop.shape[3]
    stride_h = strides[1]
    stride_w = strides[2]
    dilation_rate = 1
    padding_top, padding_bottom, padding_left, padding_right = pads

    # dilation
    dilated_height = out_height * stride_h - (stride_h - 1)
    dilated_width = out_width * stride_w - (stride_w - 1)
    if tuple(pads) == (0, 0, 0, 0):
        ori_padh = (out_height - 1) * stride_h + filter_height - input_height
        ori_padw = (out_width - 1) * stride_w + filter_width - input_width
        if ori_padh < 0:
            dilated_height -= ori_padh
        if ori_padw < 0:
            dilated_width -= ori_padw
    dilated_grad = np.zeros(
        [batch, dilated_height, dilated_width, out_channels])

    for i in range(0, out_height):
        index_h = i * stride_h
        for j in range(0, out_width):
            index_w = j * stride_w
            dilated_grad[:, index_h, index_w, :] = out_backprop[:, i, j, :]

    filter_grad_tmp = np.zeros(
        [batch, filter_height, filter_width, in_channels])
    input_piece = np.zeros([
        1, input_height + padding_top + padding_bottom,
           input_width + padding_left + padding_right, 1
    ])  # used as feature map
    dilated_grad_piece = np.zeros([dilated_height, dilated_width, 1,
                                   1])  # used as filter
    h_start = padding_top
    h_end = padding_top + input_height
    w_start = padding_left
    w_end = padding_left + input_width

    for j in range(0, batch):
        for i in range(0, in_channels):
            input_piece[0, h_start:h_end, w_start:w_end, 0] = _input[j, :, :, i]
            dilated_grad_piece[:, :, 0, 0] = dilated_grad[j, :, :, i]
            filter_grad_tmp[j, :, :, i] = _conv2D(
                input_piece, dilated_grad_piece).reshape(
                [filter_height, filter_width])
    filter_grad = np.sum(filter_grad_tmp, axis=0)

    return filter_grad


def NCHW_to_NC1C0HW(tensor):
    """
    input tensor is a 4D feature map,
    with a dimension [N, C, H, W].
    padding C to C1*C0, where C0 = 16
    output: tensor_pad [N, C1, C0, H, W]
    """
    c0 = 16
    dim = list(tensor.shape)
    padding = dim[1] % c0

    if padding != 0:
        d = dim[1]
        dim[1] = dim[1] + c0 - padding
        tensor_pad = np.zeros((dim[0], dim[1], dim[2], dim[3]))
        for i in range(dim[0]):
            tensor_pad[i, 0:d, :, :] = tensor[i, :, :, :]
    else:
        tensor_pad = tensor

    dims = [dim[0], dim[1] // c0, c0, dim[2], dim[3]]
    tensor_pad = tensor_pad.reshape(dims)

    return tensor_pad


def _conv_bp_filter(filter_ori, multi, block_size=16):
    k_c1, hk, wk, cout, _ = filter_ori.shape
    # C1HWCoC0 --> NCHW
    filter_dilate = filter_ori.transpose(3, 0, 4, 1, 2).reshape(cout, k_c1 * block_size, hk, wk)
    _filter = np.zeros((multi, k_c1 * block_size, hk, wk), dtype=filter_ori.dtype)
    for n in range(cout):
        for c in range(k_c1 * block_size):
            if n // multi == c % block_size:
                _filter[n % multi, c, :, :] = filter_dilate[n, c, :, :]
    # filter: Co_g, Ci, Hk, Wk
    return _filter


def _conv2d_bp_np(_filter, dy, dx, strides, padding, group_dict):
    batch, cout1, ho, wo, block_size = dy.shape
    cout_g, _, hk, wk = _filter.shape
    _, cin_ori, h, w = dx
    stride_h, stride_w = strides
    pad_up, pad_down, pad_left, pad_right = padding

    cin = _align(cin_ori, block_size)
    real_g = group_dict.get("real_g")
    cin_g = group_dict.get("cin_g")
    cout1_g = group_dict.get("cout1_g")

    # dy: N, Co1, Ho, Wo, Co0 -> N, Co, Ho, Wo
    dy = dy.transpose(0, 1, 4, 2, 3).reshape(
        batch, cout1 * block_size, ho, wo)
    dy_g = np.zeros([real_g, batch, cout1_g * block_size,
                     ho, wo]).astype(dy.dtype)
    for i in range(real_g):
        for j in range(batch):
            for k in range(cout_g):
                for l in range(ho):
                    for m in range(wo):
                        if i * cout1_g * block_size + k < cout1 * block_size:
                            dy_g[i, j, k, l, m] = dy[j, i *
                                                     cout1_g * block_size + k, l, m]

    # filter: Co_g, g*Ci1_g*Ci0, Hk, Wk  -> g, Co_g, Ci_g, Hk, Wk
    _filter = _filter.reshape(cout_g, real_g, cin_g, hk,
                              wk).transpose(1, 0, 2, 3, 4)

    # calculate dx
    dx_g = np.zeros([real_g, batch, cin_g, h, w])
    dx_g_pad = np.pad(dx_g, [(0, 0), (0, 0), (0, 0),
                             (pad_up, pad_down), (pad_left, pad_right)])
    for m in range(real_g):
        for n in range(batch):
            for i in range(ho):
                for j in range(wo):
                    for f in range(cout_g):
                        dx_g_pad[m, n, :, i * stride_h: i * stride_h + hk, j *
                                                                           stride_w: j * stride_w + wk] += _filter[
                                                                                                               m, f] * \
                                                                                                           dy_g[
                                                                                                               m, n, f, i, j]

    dx_g = dx_g_pad[:, :, :, pad_up: pad_up + h, pad_left: pad_left + w]
    # G, N, Ci_ori, H, W -> N, Ci, H, W -> N, Ci1, H, W, Ci0
    dx = np.zeros([batch, cin, h, w]).astype(dy.dtype)
    for j in range(cin):
        dx[:, j, :, :] = dx_g[j // cin_g, :, j % cin_g, :, :]
    return dx


# noinspection PyUnusedLocal
def _conv2d_dx_golden(out_backprop, filter_ori, input_size, cout_ori, strides, pads, dilations, groups=1,
                      out_backprop_formats='NC1HWC0', out_formats='NC1HWC0', padding=None, bias=None,
                      output_dtype='float16'):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    pad_top, pad_bottom, pad_left, pad_right = pads
    strideh, stridew = strides
    dilationh, dilationw = dilations
    block_size = 16
    k_c1, hk, wk, cout_g, _ = filter_ori.shape
    k_c = k_c1 * block_size
    Ni, cin_ori, Hi, Wi = input_size

    if out_backprop_formats == "NC1HWC0":
        batch, cout1, Ho, Wo, block_size = out_backprop.shape
        cout = cout1 * block_size
    # elif out_backprop_formats == "NCHW":
    else:
        batch, cout, Ho, Wo = out_backprop.shape
    if out_formats == "NC1HWC0":
        cin1 = _ceil(cin_ori, block_size)
        cin = cin1 * block_size
    else:
        cin = cin_ori
    ori_dy_dtype = out_backprop.dtype
    if ori_dy_dtype == 'float16':
        filter_ori = filter_ori.astype('float32')
        out_backprop = out_backprop.astype('float32')
    # if output_dtype == 'float16':
    #     bias_dtype = 'float32'
    # else:
    #     bias_dtype = output_dtype
    bias_dtype = output_dtype
    if groups == 1 and (dilationh, dilationw) == (1, 1):
        # 5HD to NHWC
        if out_backprop_formats == "NC1HWC0":
            dy = out_backprop.transpose(0, 2, 3, 1, 4).reshape(batch, Ho, Wo, cout)
        # elif out_backprop_formats == "NCHW":
        else:
            dy = out_backprop.transpose(0, 2, 3, 1).reshape(batch, Ho, Wo, cout)
        # C1HWNC0 to HWCN
        w_dilate = filter_ori.transpose(1, 2, 0, 4, 3).reshape(hk, wk, k_c, cout_g)
        w = w_dilate[:, :, :cin, :cout]
        print([Ni, Hi, Wi, cin], w.shape, dy.shape)
        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        '''
        padding, _ = _getPadding(pads, [Ni, Hi, Wi, cin], w.shape, dy.shape,
                              (strideh, stridew), [dilationh, dilationw])
        tf_conv2d_transpose = tf.compat.v1.nn.conv2d_transpose(tensor_dy, tensor_filter, [Ni, Hi, Wi, cin],
                                                     strides=[strideh, stridew],
                                                     padding=padding,
                                                     data_format="NHWC", dilations=[1, dilationh, dilationw, 1])
        '''
        tf_dx = tf.compat.v1.nn.conv2d_backprop_input([Ni, Hi, Wi, cin], tensor_filter, tensor_dy,
                                                      strides=[1, strideh, stridew, 1],
                                                      padding=((0, 0), (pad_top, pad_bottom),
                                                               (pad_left, pad_right), (0, 0)),
                                                      data_format='NHWC', use_cudnn_on_gpu=False,
                                                      dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_filter: w, tensor_dy: dy}
        tf_dx = tf.compat.v1.cast(tf_dx, dtype=tf.float16)
        if bias is not None:
            print("bias_shape", bias.shape, bias_dtype)
            bias = bias.astype(bias_dtype)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            tf_dx = tf.compat.v1.nn.bias_add(tf_dx, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(tf_dx, feed_dict=feed_dict)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx tf_dx,group=1,', out_backprop.shape, dy.shape,
              filter_ori.shape, w.shape, [Ni, Hi, Wi, cin], out.shape)
        # NHWC to NC1HWC0
        if out_formats == "NC1HWC0":
            output = out.reshape((Ni, Hi, Wi, cin1, block_size)).transpose(
                0, 3, 1, 2, 4)
        # NHWC to NCHW
        # elif out_formats == "NCHW":
        else:
            output = out.transpose(0, 3, 1, 2)
    elif groups == cin_ori and strideh == stridew and (dilationh, dilationw) == (1, 1):
        # 5HD to NHWC
        dy = out_backprop.transpose(0, 2, 3, 1, 4).reshape(batch, Ho, Wo, cout)
        # C1HWNC0 _> NCHW--> HWCN HWC1
        multi = cout_ori // cin_ori
        w = _conv_bp_filter(filter_ori, multi, block_size).transpose((2, 3, 1, 0))

        padding, _ = _getPadding(pads, [Ni, Hi, Wi, cin], w.shape, dy.shape,
                                 (strideh, stridew), [dilationh, dilationw])
        # tf.compat.v1.enable_eager_execution()
        tensor_filter = tf.compat.v1.placeholder(w.dtype, shape=w.shape)
        tensor_dy = tf.compat.v1.placeholder(dy.dtype, shape=dy.shape)
        dp_dx = tf.compat.v1.nn.depthwise_conv2d_backprop_input([Ni, Hi, Wi, cin], tensor_filter, tensor_dy,
                                                                strides=[1, strideh, stridew, 1],
                                                                padding=padding, data_format="NHWC",
                                                                dilations=[1, dilationh, dilationw, 1])
        feed_dict = {tensor_dy: dy, tensor_filter: w}
        dp_dx = tf.compat.v1.cast(dp_dx, dtype=tf.float16)
        if bias is not None:
            print("bias_shape", bias.shape, bias_dtype)
            bias = bias.astype(bias_dtype)
            tensor_bias = tf.compat.v1.placeholder(bias.dtype, shape=bias.shape)
            dp_dx = tf.compat.v1.nn.bias_add(dp_dx, tensor_bias)
            feed_dict[tensor_bias] = bias
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            out = sess.run(dp_dx, feed_dict=feed_dict)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxtf_depthwise_dx,group=cin_ori,', out_backprop.shape, dy.shape,
              filter_ori.shape, w.shape, [Ni, cin, Hi, Wi], out.shape)
        # N, H, W, Ci_ori -> N, H, W, Ci -> N, Ci1, H, W, Ci0
        # out = np.pad(out, ((0, 0), (0, 0), (0, 0), (0, cin - cin_ori)), 'constant')
        # NHWC to NC1HWC0
        output = out.reshape((Ni, Hi, Wi, cin1, block_size)
                             ).transpose(0, 3, 1, 2, 4)
    # elif pad_top == pad_bottom and pad_left == pad_right:
    #     import torch
    #     torch.backends.cudnn.benchmark = True

    #     group_dict = _calculate_group(cin_ori, cout_ori, groups)
    #     ### 5HD to NCHW
    #     dy = out_backprop.transpose(0, 1, 4, 2, 3).reshape(batch, cout, Ho, Wo)
    #     ### dy_nhwc = out_backprop.transpose(0, 2, 3, 1, 4).reshape(batch, Ho, Wo, cout)
    #     ### dy = dy_nhwc[:, :, :, :cout_ori].transpose(0, 3, 1, 2)
    #     ### C1HWNC0 to NCHW
    #     multi = cout_ori // cin_ori
    #     w = conv_bp_filter(filter_ori, multi, block_size).reshape(cout, 1, hk, wk)
    #     ### w = conv_group_filter(filter_ori, group_dict)
    #     if bias is not None:
    #         print("bias_shape", bias.shape, bias_dtype)
    #         pytorch_bias = torch.from_numpy(bias.astype(bias_dtype))
    #     else:
    #         pytorch_bias = None
    #     print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx', out_backprop.shape, dy.shape,  filter_ori.shape, w.shape)
    #     pytorch_dy = torch.from_numpy(dy) # .type(torch.float32)
    #     pytorch_w = torch.from_numpy(w)
    #     print('xxxxxxxxxxxtorch_dx:', [Ni, cin, Hi, Wi],
    #         filter_ori.shape, pytorch_w.shape, pytorch_dy.shape, groups)
    #     out = torch.nn.functional.conv_transpose2d(
    #         pytorch_dy, pytorch_w, bias=pytorch_bias, stride=[strideh, stridew],
    #         padding=[pad_top, pad_left], output_padding=0, groups=cin,
    #         dilation=[dilationh, dilationw])
    #     out = out.numpy()
    #     print('xxxxxxxxxxxtorch_dx:',  out.shape, [Ni, cin, Hi, Wi])
    #     ###out = np.pad(out, ((0, 0), (0, cin - cin_ori), (0, 0), (0, 0)), 'constant')
    #     ### NCHW to NC1HWC0
    #     output = out.reshape((Ni, cin1, block_size, Hi, Wi)).transpose(0, 1, 3, 4, 2)
    else:
        group_dict = _calculate_group(cin_ori, cout_ori, groups)
        # C1HWNC0 --> NCHW
        filter_dilate = filter_ori.transpose(3, 0, 4, 1, 2).reshape(
            cout_g, k_c1 * block_size, hk, wk)
        output = _conv2d_bp_np(filter_dilate, out_backprop, [Ni, cin, Hi, Wi], [
            strideh, stridew], pads, group_dict)
        output.astype(output_dtype)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxnumpy,other,', out_backprop.shape, filter_ori.shape, filter_dilate.shape,
              [Ni, cin, Hi, Wi], output.shape)
        if bias is not None:
            print("bias_shape", bias.shape, bias_dtype)
            bias = bias.astype(bias_dtype)
            output = output + bias[np.newaxis, :, np.newaxis, np.newaxis]
        output = output.reshape((Ni, cin1, block_size, Hi, Wi)
                                ).transpose(0, 1, 3, 4, 2)

    return output


# noinspection PyUnusedLocal
def _conv2D(input_, filter_, strides=(1, 1, 1, 1), padding=None):
    ish = input_.shape
    fsh = filter_.shape
    output = np.zeros([ish[0], (ish[1] - fsh[0]) // strides[1] + 1, (ish[2] - fsh[1]) // strides[2] + 1, fsh[3]])
    osh = output.shape

    for p in range(osh[0]):
        for i in range(osh[1]):
            for j in range(osh[2]):
                for di in range(fsh[0]):
                    for dj in range(fsh[1]):
                        t = np.dot(
                            input_[p, strides[1] * i + di, strides[2] * j + dj, :], filter_[di, dj, :, :])
                        output[p, i, j] = np.sum([t, output[p, i, j]], axis=0)
    return output


def _conv2d_bp_common(filter_ori, dy_ori, group_dict, multi, cin_ori, block_size):
    real_g = group_dict.get("real_g")
    cin_g = group_dict.get("cin_g")
    batch, cout1, ho, wo, _ = dy_ori.shape
    if len(filter_ori.shape) == 4:
        hk, wk, cout_g, _ = filter_ori.shape
        # filter: Hk, Wk, Co_g, Ci0 -> Co_g, Ci0, Hk, Wk
        filter_dilate = filter_ori.transpose(2, 3, 0, 1).reshape(
            cout_g, real_g * cin_g, hk, wk).astype(filter_ori.dtype)
    elif len(filter_ori.shape) == 5:
        _, hk, wk, cout_g, block_size = filter_ori.shape
        # filter: g*Ci1_g, Hk, Wk, Co_g, Ci0 -> Co_g, g*Ci1_g*Ci0, Hk, Wk
        filter_dilate = filter_ori.transpose(3, 0, 4, 1, 2).reshape(
            cout_g, real_g * cin_g, hk, wk).astype(filter_ori.dtype)
    else:
        raise RuntimeError(
            f"dim of filter_ori's shape is invalid, support 4 or 5, now is {len(filter_ori.shape)}!")

    # filter: Co_g, Ci, Hk, Wk -> K, Ci, Hk, Wk
    _filter = np.zeros((multi, cin_ori, hk, wk)).astype(filter_ori.dtype)
    for n in range(cout_g):
        for c in range(cin_ori):
            if n // multi == c % block_size:
                _filter[n % multi, c, :, :] = filter_dilate[n, c, :, :]
    # filter: Co_g, Ci, Hk, Wk -> Hk, Wk, Ci, K
    _filter = _filter.transpose((2, 3, 1, 0)).astype(np.float32)
    # dy: N, Co1, Ho, Wo, Co0 -> N, Ho, Wo, Co
    dy = dy_ori.transpose((0, 2, 3, 1, 4)).reshape((batch, ho, wo, cout1 * block_size)).astype(np.float32)
    # dy: N, Ho, Wo, Co -> N, Ho, Wo, Co_ori
    dy = dy[:, :, :, :multi * cin_ori]

    return filter_dilate, _filter, dy

