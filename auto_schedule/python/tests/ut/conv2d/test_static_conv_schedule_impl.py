#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import traceback
from sch_test_frame.ut import OpUT
import te
import tbe
from te import tvm
from impl.util.platform_adapter import operation
from impl.conv2d import conv2d_compute
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from impl.relu_v2 import relu_v2_compute
from tbe.dsl.compute.conv_compute import ConvParam

ut_case = OpUT("conv_schedule", "conv_schedule.test_static_conv_schedule_impl")
def test_bias_preload(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 1, 32, 32, 32], name="x", dtype="int8", attrs={"ori_shape": [1, 1, 32, 32], "format": "NC1HWC0", "ori_format": "NCHW"})
                filter_ = tvm.placeholder([9, 4, 16, 32], name="filter", dtype="int8", attrs={"ori_shape": [64, 1, 3, 3], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([64], name="bias", dtype="int32", attrs={"ori_shape": [64], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "int32", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 2, 2]
                pads = [0, 1, 0, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
                scale_dtype = "float16"
                v200_version = ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
                if te.platform.cce_conf.get_soc_spec("SOC_VERSION") in v200_version:
                    scale_dtype = "uint64"
                deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [64]})
                dequant_out = ascend_dequant_compute(conv_out, deq_scale, None)
                tensor_list = [x_, filter_, bias_, deq_scale, dequant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(dequant_out)
                config = {
                    "name": "bias_preload_910A",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_bias_preload")
ut_case.add_cust_test_func("Ascend910A", test_func=test_bias_preload)

def test_conv2d_reluv2_mask_buffer_align(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 4, 300, 450, 16], name="x", dtype="float16", attrs={"ori_shape": [1, 300, 450, 64], "format": "NC1HWC0", "ori_format": "NHWC"})
                filter_ = tvm.placeholder([36, 8, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [3, 3, 64, 128], "format": "FRACTAL_Z", "ori_format": "HWCN"})
                bias_ = tvm.placeholder([128], name="bias", dtype="float16", attrs={"ori_shape": [128], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
                strides = [1, 1, 1, 1]
                pads = [1, 1, 1, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NHWC")
                data_res, mask = relu_v2_compute(conv_out, None, None)
                tensor_list = [x_, filter_, bias_, data_res, mask]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule([data_res, mask])
                config = {
                    "name": "conv2d_reluv2_mask_buffer_alig_910A",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_conv2d_reluv2_mask_buffer_align")
ut_case.add_cust_test_func("Ascend910A", test_func=test_conv2d_reluv2_mask_buffer_align)

def test_static_conv_schedule_quant_fusion(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 4, 300, 450, 16], name="x", dtype="float16", attrs={"ori_shape": [1, 300, 450, 64], "format": "NC1HWC0", "ori_format": "NHWC"})
                filter_ = tvm.placeholder([36, 8, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [3, 3, 64, 128], "format": "FRACTAL_Z", "ori_format": "HWCN"})
                bias_ = tvm.placeholder([128], name="bias", dtype="float16", attrs={"ori_shape": [128], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
                strides = [1, 1, 1, 1]
                pads = [1, 1, 1, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NHWC")
                quant_out = ascend_quant_compute(conv_out, None, scale=0.1, offset=0.2, sqrt_mode=True)
                tensor_list = [x_, filter_, bias_, quant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(quant_out)
                config = {
                    "name": "static_conv_schedule_quant_fusion_910A",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_static_conv_schedule_quant_fusion")
ut_case.add_cust_test_func("Ascend910A", test_func=test_static_conv_schedule_quant_fusion)


def test_bias_preload(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 1, 32, 32, 32], name="x", dtype="int8", attrs={"ori_shape": [1, 1, 32, 32], "format": "NC1HWC0", "ori_format": "NCHW"})
                filter_ = tvm.placeholder([9, 4, 16, 32], name="filter", dtype="int8", attrs={"ori_shape": [64, 1, 3, 3], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([64], name="bias", dtype="int32", attrs={"ori_shape": [64], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "int32", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 2, 2]
                pads = [0, 1, 0, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
                scale_dtype = "float16"
                v200_version = ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
                if te.platform.cce_conf.get_soc_spec("SOC_VERSION") in v200_version:
                    scale_dtype = "uint64"
                deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [64]})
                dequant_out = ascend_dequant_compute(conv_out, deq_scale, None)
                tensor_list = [x_, filter_, bias_, deq_scale, dequant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(dequant_out)
                config = {
                    "name": "bias_preload_710",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_bias_preload")
ut_case.add_cust_test_func("Ascend710", test_func=test_bias_preload)

def test_conv2d_reluv2_mask_buffer_align(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 4, 300, 450, 16], name="x", dtype="float16", attrs={"ori_shape": [1, 300, 450, 64], "format": "NC1HWC0", "ori_format": "NHWC"})
                filter_ = tvm.placeholder([36, 8, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [3, 3, 64, 128], "format": "FRACTAL_Z", "ori_format": "HWCN"})
                bias_ = tvm.placeholder([128], name="bias", dtype="float16", attrs={"ori_shape": [128], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
                strides = [1, 1, 1, 1]
                pads = [1, 1, 1, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NHWC")
                data_res, mask = relu_v2_compute(conv_out, None, None)
                tensor_list = [x_, filter_, bias_, data_res, mask]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule([data_res, mask])
                config = {
                    "name": "conv2d_reluv2_mask_buffer_alig_710",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_conv2d_reluv2_mask_buffer_align")
ut_case.add_cust_test_func("Ascend710", test_func=test_conv2d_reluv2_mask_buffer_align)

def test_static_conv_schedule_quant_fusion(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 4, 300, 450, 16], name="x", dtype="float16", attrs={"ori_shape": [1, 300, 450, 64], "format": "NC1HWC0", "ori_format": "NHWC"})
                filter_ = tvm.placeholder([36, 8, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [3, 3, 64, 128], "format": "FRACTAL_Z", "ori_format": "HWCN"})
                bias_ = tvm.placeholder([128], name="bias", dtype="float16", attrs={"ori_shape": [128], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
                strides = [1, 1, 1, 1]
                pads = [1, 1, 1, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NHWC")
                quant_out = ascend_quant_compute(conv_out, None, scale=0.1, offset=0.2, sqrt_mode=True)
                tensor_list = [x_, filter_, bias_, quant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(quant_out)
                config = {
                    "name": "static_conv_schedule_quant_fusion_710",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_static_conv_schedule_quant_fusion")
ut_case.add_cust_test_func("Ascend710", test_func=test_static_conv_schedule_quant_fusion)

def test_static_conv_schedule_blo_buffer_tile(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 23, 4, 4, 16], name="x", dtype="float16", attrs={"ori_shape": [1, 360, 4, 4,], "format": "NC1HWC0", "ori_format": "NCHW"})
                filter_ = tvm.placeholder([23, 23, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [120, 360, 1, 1], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([360], name="bias", dtype="float16", attrs={"ori_shape": [360], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NHWC"}
                strides = [1, 1, 1, 1]
                pads = [0, 0, 0, 0]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NHWC")
                tensor_list = [x_, filter_, bias_, conv_out]
                tiling = {'AL0_matrix':[1, 1, 16, 16], 'CL0_matrix': [1, 1, 16, 16, 1], 'CUB_matrix': [1, 1, 16, 16], 
                    'A_overhead_opt_flag': 0, 'B_overhead_opt_flag': 0, 'BL0_matrix': [ ],
                    'manual_pingpong_buffer': {'AL0_pbuffer': 1, 'AL1_pbuffer': 1, 'AUB_pbuffer': 1, 'BL0_pbuffer': 1, 
                    'BL1_pbuffer': 1, 'BUB_pbuffer': 1, 'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 1},
                    'n_bef_batch_flag': 0, 'AL1_shape': [], 'BL1_shape': None, 'block_dim': [1, 8, 1, 1], 'CUB_channel_wise_flag': False}
                ConvParam.tiling = tiling
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(conv_out)
                config = {
                    "name": "static_conv_schedule_quant_fusion",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }

                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding static_conv_schedule_blo_buffer_tile")
ut_case.add_cust_test_func("Ascend710",test_func=test_static_conv_schedule_blo_buffer_tile)


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])