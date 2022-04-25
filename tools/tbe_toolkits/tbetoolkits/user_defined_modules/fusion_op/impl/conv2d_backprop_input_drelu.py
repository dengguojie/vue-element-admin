# -*- coding:utf-8 -*-
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d
from impl.conv2d_backprop_input_d import conv2d_backprop_input_d_compute
from impl.relu_grad_v2 import relu_grad_v2_compute
from impl.add_n import add_n_compute_for_fusion as add_n_compute
from te.tvm import api as tvm
from te.utils.cce import auto_schedule
from te.tvm.target import cce
from te.utils.cce import auto_schedule
from te.lang.cce import cce_build_code as build
from impl.util import util_deconv_comm

def conv2d_backprop_input_drelu(filter, out_backprop, y, input_size, strides, pads, dilations=(1, 1, 1, 1),
                                groups=1, data_format="NHWC", kernel_name="conv2d_backprop_input"):
    DIM_OUT_N = out_backprop.get('ori_format').index('N')
    DIM_OUT_H = out_backprop.get('ori_format').index('H')
    DIM_OUT_W = out_backprop.get('ori_format').index('W')
    DIM_OUT_C = out_backprop.get('ori_format').index('C')
    DIM_IN_N = data_format.index('N')
    DIM_IN_H = data_format.index('H')
    DIM_IN_W = data_format.index('W')
    DIM_IN_C = data_format.index('C')
    DIM_F_N = filter.get('ori_format').index('N')
    DIM_F_C = filter.get('ori_format').index('C')
    DIM_F_H = filter.get('ori_format').index('H')
    DIM_F_W = filter.get('ori_format').index('W')
    out_shape = out_backprop.get('ori_shape')
    filter_shape = filter.get('ori_shape')
    dedy_batch, dedy_channel, dedy_h, dedy_w = out_shape[DIM_OUT_N], out_shape[DIM_OUT_C], \
        out_shape[DIM_OUT_H], out_shape[DIM_OUT_W]
    filter_batch, filter_channel, filter_h, filter_w = filter_shape[DIM_F_N], \
        filter_shape[DIM_F_C], filter_shape[DIM_F_H], filter_shape[DIM_F_W]

    def _ceil(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2

    def _align(x_1, x_2):
        if x_2 == 0:
            raise RuntimeError("Division by zero")
        return (x_1 + x_2 - 1) // x_2 * x_2

    c0_size = 16  # Channel axis should be align with 16
    shape_dedy = (dedy_batch, _ceil(dedy_channel, c0_size), dedy_h, dedy_w, c0_size)
    nchw_out_shape = (out_shape[DIM_OUT_N], out_shape[DIM_OUT_C], out_shape[DIM_OUT_H], out_shape[DIM_OUT_W])
    nchw_input_size = (input_size[DIM_IN_N], input_size[DIM_IN_C], input_size[DIM_IN_H], input_size[DIM_IN_W])
    nchw_filter_shape = (filter_shape[DIM_F_N], filter_shape[DIM_F_C], filter_shape[DIM_F_H], filter_shape[DIM_F_W])
    group_dict = util_deconv_comm.calculate_group(
        nchw_out_shape,
        nchw_input_size,
        nchw_filter_shape,
        groups,
        filter.get('dtype'),
        'NCHW'
    )
    g_extend = group_dict.get(util_deconv_comm.GroupDictKeys.g_extend)
    dx_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dx_c1_extend)
    dy_c1_extend = group_dict.get(util_deconv_comm.GroupDictKeys.dy_c1_extend)
    shape_filter_frac = (# (GCi1HkWk, Co1, Co0, Ci0); filter_placehold is same to conv2d_forward's filter
                         g_extend * dx_c1_extend * filter_h * filter_w,
                         dy_c1_extend,
                         c0_size,
                         c0_size,
                        )
    dedy = tvm.placeholder(shape_dedy, name="dedy", dtype="float16",
                           attrs={"ori_shape": out_backprop.get('ori_shape'), "dtype": "float16", 
                                  "ori_format": out_backprop.get('ori_format')})
    # pylint: disable=redefined-builtin
    filter = tvm.placeholder(shape_filter_frac, name="filter", dtype="float16",
                             attrs={"ori_shape": filter.get('ori_shape'), "dtype": "float16", 
                                    "ori_format": filter.get('ori_format')})
    dx = {"ori_shape": input_size, "dtype": "float16", "ori_format": data_format}

    dedx = conv2d_backprop_input_d_compute(filter, dedy, dx, input_size, strides,
                                           pads, dilations, groups, data_format, kernel_name)

    mask_shape = [i.value for i in dedx.shape]
    mask = tvm.placeholder(mask_shape, name="mask", dtype='bool')
    res = relu_grad_v2_compute(dedx, mask, {}, kernel_name="relu_grad_v2")
    tensor_list = [filter, dedy, mask, res]
    with cce():
        sch = auto_schedule(res)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": kernel_name,
        "tensor_list": tensor_list
    }

    build(sch, config)
