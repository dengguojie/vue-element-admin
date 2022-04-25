# -*- coding:utf-8 -*-

import tbe
from te.platform.cce_conf import te_set_version
import te
from tbe import tvm
import tbe.dsl.base as tbe_base
from tbe.dsl.base import operation
from tbe.common.register import register_operator
from tbe.dsl import build

from impl.dynamic.conv2d_backprop_filter import conv2d_backprop_filter_fusion_compute
from impl.dynamic.trans_data import trans_data_fusion_compute


@register_operator('Conv2DBackpropFilter')
def conv2d_bp_filter_transdata(fmap,
                              filter_size, out_backprop, y, strides,
                              pads, dilations=(1, 1, 1, 1),
                              groups=1, data_format="NHWC",
                              kernel_name="conv2d_backprop_filter"):
    print("======== enter conv2d_bp_filter_transdata ========")
    with tbe.dsl.base.operation.compute():
        fmap_dtype = fmap.get("dtype", "float16").lower()
        dedy_dtype = out_backprop.get("dtype", "float16").lower()

        batch = operation.var("batch")
        fmap_c = operation.var("fmap_c")
        fmap_h = operation.var("fmap_h")
        fmap_w = operation.var("fmap_w")

        dedy_c = operation.var("dedy_c")
        dedy_h = operation.var("dedy_h")
        dedy_w = operation.var("dedy_w")

        fmap_nchw = (batch, fmap_c, fmap_h, fmap_w)
        dedy_nchw = (batch, dedy_c, dedy_h, dedy_w)

        fmap_nc_hw = (batch, fmap_c, fmap_h * fmap_w)
        dedy_nc_hw = (batch, dedy_c, dedy_h * dedy_w)

        fmap = tvm.placeholder(fmap_nc_hw, name="fmap", dtype=fmap_dtype, attrs = {"shape": fmap_nchw})
        dedy = tvm.placeholder(dedy_nc_hw, name="dedy", dtype=dedy_dtype, attrs = {"shape": dedy_nchw})
        filter_tensor = tvm.placeholder([4], name="filter_tensor", dtype="int32")
        y = {"shape": (-1, -1, -1, -1), "dtype": "float32", "format": "NCHW"}

        nc1hwc0_fmap = {"ori_format": "NCHW", "ori_shape": fmap_nchw,"shape":fmap_nchw, "format": "NCHW", "dtype": "float16"}
        nc1hwc0_dedy = {"ori_format": "NCHW", "ori_shape": dedy_nchw, "shape":dedy_nchw,"format": "NCHW", "dtype": "float16"}
        fmap_5hd = trans_data_fusion_compute(fmap, nc1hwc0_fmap, "NCHW", "NC1HWC0")
        dedy_5hd = trans_data_fusion_compute(dedy, nc1hwc0_dedy, "NCHW", "NC1HWC0")

        dedw = conv2d_backprop_filter_fusion_compute(fmap_5hd, filter_tensor, dedy_5hd,
            y, strides, pads, dilations, groups, data_format, kernel_name)

        with tvm.target.cce():
            sch = te.utils.cce.auto_schedule(dedw)

        tensor_list = [fmap, filter_tensor, dedy, dedw]

        config = {"name": kernel_name,
                "tensor_list": tensor_list,
                "build_args": {"constant_realize_extent_in_infer_bound": False}
                }

        build(sch, config)
