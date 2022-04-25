# -*- coding:utf-8 -*-

import tbe
from impl.dynamic.conv2d_backprop_input import conv2dbp_input_fusion_compute
from impl.dynamic.trans_data import trans_data_fusion_compute
from te.platform.cce_conf import te_set_version
import te
from tbe import tvm
import tbe.dsl.base as tbe_base
from impl.util.util_cube_dynamic import Conv2dParaProcess
from tbe.dsl.base import operation
from tbe.common.register import register_operator
from tbe.dsl import build
import tbe


def icd(num_a, num_b):
    """
    upper division
    """
    return (num_a + num_b - 1) // num_b


def lcm(wout, factor):
    """
    get least common multiple of wout and factor
    """
    tmp = wout * factor
    while wout % factor != 0:
        wout, factor = factor, (wout % factor)
    return tmp // factor

def  _conv2d_bp_input_transdata(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                                filter, out_backprop, y, strides,
                                pads, dilations=(1, 1, 1, 1),
                                groups=1, data_format="NHWC",
                                kernel_name="conv2d_backprop_input"):
    with tbe.dsl.base.operation.compute():
        dedy_range = out_backprop.get('range')
        dedx_range = y.get('range')
        filter_range = filter.get('range')
        if out_backprop.get("format") == "NCHW" or out_backprop.get("format") == "NC1HWC0":
            n_dim = 0
            c1_dim = 1
            h_dim = 2
            w_dim = 3
            fractal_z_fir_dim = 0
            fractal_z_sec_dim = 1
            dynamic_flag = -1
            unknown_flag = -2
        else:
            print("input format only is support NCHW or NC1HWC0")
            raise RuntimeError("unsupport format")
        if out_backprop.get("format") == "NCHW":
            var_n = operation.var("batch_n")
            var_dedy_c = operation.var("dedy_c")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
        else:
            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
            var_n = operation.var("batch_n")
            var_dedy_c1 = operation.var("dedy_c1")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            dy_shape_nc1hwc0 = (var_n, var_dedy_c1, var_dedy_h, var_dedy_w, 16)
            dy_tensor = tvm.placeholder(dy_shape_nc1hwc0, name="dedy", dtype="float16")

        filter_shape_frac_z = (var_filter_ci1hw, var_filter_co1, 16, 16)
        input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
        filter_tensor = tvm.placeholder(filter_shape_frac_z, name="filter", dtype="float16")
        if out_backprop.get("format") == "NCHW":
            dy_shape_nchwc = (var_n, var_dedy_c, var_dedy_h, var_dedy_w)
            transdata_in_tensor = tvm.placeholder(dy_shape_nchwc, name="transdata_in", dtype="float16")
            dst = {}
            dy_tensor = trans_data_fusion_compute(transdata_in_tensor, dst, "NCHW", "NC1HWC0")

        conv_res = conv2dbp_input_fusion_compute(input_tensor,
                                  filter_tensor, dy_tensor, y, strides, pads, dilations=(1, 1, 1, 1),
                                  groups=1, data_format='NCHW', kernel_name=kernel_name)
        # src, dst, src_format, dst_format, groups=1, kernel_name='transdata'
        trans_data_res = trans_data_fusion_compute(conv_res, {'ori_shape':y.get('ori_shape')}, "NC1HWC0", "NCHW")
        with tvm.target.cce():
            sch = te.utils.cce.auto_schedule(trans_data_res)
        if out_backprop.get("format") == "NC1HWC0":
            tensor_list = [input_tensor, filter_tensor, dy_tensor, trans_data_res]
        else:
            tensor_list = [input_tensor, filter_tensor, transdata_in_tensor, trans_data_res]

        config = {"name": kernel_name,
            "tensor_list": tensor_list,
            "build_args": {"constant_realize_extent_in_infer_bound": False,
                           "enable_branch_eliminator_else_case": False}}
        build(sch, config)


@register_operator('Conv2DBackpropInput')
def conv2d_bp_input_transdata(input_size,  # pylint: disable=W0622,C0103,R0913,R0914
                              filter, out_backprop, y, strides,
                              pads, dilations=(1, 1, 1, 1),
                              groups=1, data_format="NHWC",
                              kernel_name="conv2d_backprop_input"):
    print("enter conv2d_bp_input_transdata ========")
    _conv2d_bp_input_transdata(input_size, filter, out_backprop, y, strides,pads, dilations, groups, data_format, kernel_name)
