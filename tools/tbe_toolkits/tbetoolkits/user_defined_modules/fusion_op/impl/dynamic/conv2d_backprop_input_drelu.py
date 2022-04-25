# -*- coding:utf-8 -*-
import te
from te import tvm
from te.platform.cce_params import CUBE_MKN
from te.tvm.target import cce
from te.utils.cce import auto_schedule
from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input
from impl.dynamic.relu_grad_v2 import relu_grad_v2
from impl.dynamic.add_n import add_n
from impl.util import fusion_util
from te.platform import cce_conf
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe
from impl.dynamic.conv2d_backprop_input import conv2dbp_input_fusion_compute
from impl.dynamic.relu_grad_v2 import relu_grad_v2_compute
from tbe.dsl.base import operation

def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = [0 for i in range(len(shape))]
    j = 0
    for i in shape:
        if isinstance(i, tvm.expr.IntImm):
            tmp[j] = i.value
        else:
            tmp[j] = i

        j += 1
    return tmp

def ceil(x_1, x_2):
    """
    Get (x_1 + x_2 - 1) // x_2
    :param x_1:
    :param x_2:
    :return: (x_1 + x_2 - 1) // x_2
    """
    if x_2 == 0:
        args_dict = {
            "errCode": "E60114",
            "reason": "Division by zero",
            "value": "x_1 = {}, x_2 = {}".format(x_1, x_2),
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return (x_1 + x_2 - 1) // x_2


def _lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2

@register_operator('Conv2DBackpropInput')
def conv2d_backprop_input_drelu(input_size, filter, out_backprop, y, strides, pads, dilations=(1, 1, 1, 1), groups=1,
                                data_format='NHWC', kernel_name='conv2d_backprop_input'):
    dedy_range = out_backprop.get('range')
    filter_range = filter.get('range')
    if out_backprop.get("ori_format") == "NCHW" and out_backprop.get("format") == "NC1HWC0":
        n_dim = 0
        c1_dim = 1
        h_dim = 2
        w_dim = 3
        dynamic_flag = -1
        unknown_flag = -2
    else:
        print("input format is not NCHW,please change to NCHW")
        raise RuntimeError("unsupport format")

    # dy_shape_nc1hwc0 = list(out_backprop.get("shape"))
    # if dy_shape_nc1hwc0[n_dim] == dynamic_flag:
    #     dy_shape_nc1hwc0[n_dim] = operation.var("batch_n", dedy_range[n_dim])
    # if dy_shape_nc1hwc0[c1_dim] == dynamic_flag:
    #     dy_shape_nc1hwc0[c1_dim] = operation.var("dedy_c1", dedy_range[c1_dim])
    # if dy_shape_nc1hwc0[h_dim] == dynamic_flag:
    #     dy_shape_nc1hwc0[h_dim] = operation.var("dedy_h", dedy_range[h_dim])
    # if dy_shape_nc1hwc0[w_dim] == dynamic_flag:
    #     dy_shape_nc1hwc0[w_dim] = operation.var("dedy_w", dedy_range[w_dim])
    #
    # filter_shape_frac_z = list(filter.get("shape"))
    if filter.get("format") == "NC1HWC0":
        filter["format"] = "FRACTAL_Z"
        dx_c_ori = filter["ori_shape"][filter["ori_format"].find("C")]
        dy_c_ori = filter["ori_shape"][filter["ori_format"].find("N")] // groups
        dx_c_extend = _lcm(dx_c_ori, 16) // dx_c_ori
        dy_c_extend = _lcm(dy_c_ori, 16) // dy_c_ori
        multiple_extend = min(_lcm(dx_c_extend, dy_c_extend), groups)
        g_extend = ceil(groups, multiple_extend)
        dx_c1_extend = ceil(multiple_extend * dx_c_ori, 16)
        dy_c1_extend = ceil(multiple_extend * dy_c_ori, 16)
        filter_shape_frac_z_first = g_extend * dx_c1_extend * filter["ori_shape"][filter["ori_format"].find("H")] * filter["ori_shape"][filter["ori_format"].find("W")]
        filter_shape_frac_z_second = dy_c1_extend
        filter["shape"] = [filter_shape_frac_z_first, filter_shape_frac_z_second, 16, 16]
    # input_size = tvm.placeholder([4], name="input_size", dtype="int32",attrs=input_size)
    # out_backprop = tvm.placeholder(dy_shape_nc1hwc0, name="dedy", dtype="float16",attrs=out_backprop)
    # filter = tvm.placeholder(filter_shape_frac_z, name="filter", dtype="float16",attrs=filter)
    # print("filter",filter)
    # print("filter",type(filter.op.attrs))

    with tbe.compute():
        conv2dbp_input_info = conv2dbp_input_fusion_compute(input_size, filter, out_backprop, y, strides,
                                                            pads, dilations, groups, data_format, kernel_name)

        conv2dbp_input_res = conv2dbp_input_info.get("op_res")[0]

        mask_shape = shape_to_list(conv2dbp_input_res.shape)
        mask = tvm.placeholder(mask_shape, name="mask", dtype='uint1')

        relu_grad_v2_res = relu_grad_v2_compute(conv2dbp_input_res, mask, {}, "relu_grad_v2")
        print("conv2dbp_input_info",conv2dbp_input_info)
        print("conv2dbp_input_res",conv2dbp_input_res)
        print("mask_shape",mask_shape)
        print("mask",mask)
        print("relu_grad_v2_res",relu_grad_v2_res)
        with cce():
            sch = auto_schedule(relu_grad_v2_res)
        tensor_list = list(conv2dbp_input_info['op_placeholder']) + [mask, relu_grad_v2_res]
        config = {"name": kernel_name,
                  "tensor_list": tensor_list,
                  "build_args": {"constant_realize_extent_in_infer_bound": False}}
    tbe.build(sch, config)