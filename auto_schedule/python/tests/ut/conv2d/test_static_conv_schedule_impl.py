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


def test_aipp_conv2d(config_dict):
    """
    c04 + aipp
    """
    import json
    from impl.aipp import aipp_compute

    casename, dataflow, conv_type, shape_in, shape_w, pads, strides, dilation, groups, bias_flag, \
    quant_scale, quant_offset, relu_param = config_dict

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci0 = 4
    Co0 = 16
    Ci1 = 1
    Co1 = (Co + Co0 - 1) // Co0

    shape_w_fracz = ((Hk * Wk * Ci1 + Ci0 - 1) // Ci0, Co1, Co0, Ci0 * 4)
    weight_format = "FRACTAL_Z_C04"

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    aipp_format_dict = {
        "yuv": "YUV420SP_U8",
        "xrgb": "XRGB8888_U8",
        "rgb": "RGB888_U8"
    }
    aipp_format = "rgb"
    aipp_input_format = aipp_format_dict[aipp_format]

    h_after_crop = Hi
    w_after_crop = Wi
    output_data = {
        'shape': [Ni, 1, h_after_crop, w_after_crop, 4],
        'ori_shape': [Ni, 3, h_after_crop, w_after_crop],
        'format': 'NC1HWC0_C04',
        'ori_format': 'NCHW',
        'dtype': 'float16',
        'addr_type': 0,
        'valid_shape': (),
        'slice_offset': (),
        'use_L1_workspace': 0,
        'L1_workspace_size': -1,
        'L1_fusion_type': -1,
        'L1_addr_offset': 0,
        'total_shape': (),
        'split_index': 0
    }

    aipp_config_dict = {
        "cce_product": "2.1",
        "out_dtype": "float16",
        "out_format": "NC1HWC0",
        "aipp_mode": "static",
        "related_input_rank": 0,
        "input_format": aipp_input_format,
        "src_image_size_n": shape_in[0],
        "src_image_size_c": shape_in[1],
        "src_image_size_h": shape_in[2],
        "src_image_size_w": shape_in[3],
        "crop": False,
        "load_start_pos_h": 2,
        "load_start_pos_w": 2,
        "crop_size_h": h_after_crop,
        "crop_size_w": w_after_crop,
        "src_image_h": shape_in[2],
        "src_image_w": shape_in[3],
        "resize": 0,
        "resize_model": 0,
        "resize_output_h": 32,
        "resize_output_w": 32,
        "padding": 0,
        "left_padding_size": 7,
        "right_padding_size": 7,
        "top_padding_size": 0,
        "bottom_padding_size": 4,
        "csc_switch": 1,
        "rbuv_swap_switch": 0,
        "ax_swap_switch": 0,
        "matrix_r0c0": 298,
        "matrix_r0c1": 516,
        "matrix_r0c2": 0,
        "matrix_r1c0": 298,
        "matrix_r1c1": -100,
        "matrix_r1c2": -208,
        "matrix_r2c0": 298,
        "matrix_r2c1": 0,
        "matrix_r2c2": 409,
        "input_bias_0": 16,
        "input_bias_1": 128,
        "input_bias_2": 128,
        "mean_chn_0": 1,
        "mean_chn_1": 1,
        "mean_chn_2": 1,
        "mean_chn_3": 1,
        "var_reci_chn_0": 1.0,
        "var_reci_chn_1": 1.0,
        "var_reci_chn_2": 1.0,
        "min_chn_0": 0,
        "min_chn_1": 0,
        "min_chn_2": 0,
        "min_chn_3": 0
    }

    with tvm.target.cce():
        fmap = tvm.placeholder(shape_in,
                               name="params_0",
                               dtype="uint8",
                               attrs={
                                   "ori_shape": shape_in,
                                   "format": "NCHW",
                                   "ori_format": "NHWC"
                               })
        aipp_config_dict_json = json.dumps(aipp_config_dict)
        aipp_res = aipp_compute(fmap,
                                None,
                                output_data,
                                aipp_config_dict_json,
                                kernel_name="aipp")
        aipp_res.op.attrs["is_first_layer"] = True
        weight = tvm.placeholder(shape_w_fracz,
                                 name='weight',
                                 dtype=conv_type,
                                 attrs={
                                     'ori_shape': shape_w,
                                     'ori_format': "NCHW",
                                     'format': weight_format
                                 })
        conv_res = conv2d_compute(aipp_res,
                                  weight,
                                  None,
                                  None,
                                  None,
                                  strides,
                                  pads,
                                  dilations,
                                  offset_x=0)
        sch = te.utils.cce.auto_schedule(conv_res)


def run_v300_batch_cases_aipp(case_list, is_hf32_flag=False):
    from te.platform.cce_conf import te_set_version
    from tbe.common.platform.platform_info import get_soc_spec
    from tbe.common.context import op_context
    with op_context.OpContext():
        te_set_version("Ascend320", "AiCore")
        for case in case_list:
            test_aipp_conv2d(case)


def run_v300_aipp_fusion_cases(test_arg):
    aipp_conv2d = [
        ("conv2d", "conv2d", "float16", (1, 3, 32, 32), (32, 3, 3, 3), (1, 1, 1, 1), (1, 1), 1, 1,
         False, 0, 0, 0),
    ]
    try:
        run_v300_batch_cases_aipp(aipp_conv2d)
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
ut_case.add_cust_test_func(test_func=test_static_conv_schedule_quant_fusion)
ut_case.add_cust_test_func(test_func=run_v300_aipp_fusion_cases)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])
