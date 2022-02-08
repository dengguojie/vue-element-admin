#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.conv2d_backprop_input import conv2d_backprop_input_generalization
from te import tvm
from tbe.dsl.base import operation
from tbe.dsl import auto_schedule
from tbe.dsl import build
from tbe.common.context.op_context import OpContext
from impl.dynamic.conv2d_backprop_input import conv2dbp_input_fusion_compute
from impl.dynamic.trans_data import trans_data_fusion_compute

def test_conv2d_backprop_input_fuzz_build_lower_limit():
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (16, 3, 16, 16)
        }, {
            'ori_shape': (33, 3, 3, 5),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 3, -1, -1, 16),
            'ori_shape': (-1, 33, -1, -1),
            'range': ((16, 31), (3, 3), (4, 15), (4, 15), (16, 16)),
            'ori_range': ((16, 31), (33, 33), (4, 15), (4, 15)),
            'ori_format': 'NHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 3, -1, -1),
            'range': ((16, 31), (1, 1), (16, 31), (16, 31), (16, 16)),
            'ori_range': ((16, 31), (3, 3), (16, 31), (16, 31)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

def test_conv2d_backprop_input_fuzz_build_upper_limit():
    input_list = [
        {
            'shape': (4,),
            'ori_shape': (4,),
            'ori_format': 'NCHW',
            'format': 'NCHW',
            'dtype': 'int32',
            'const_value': (50, 2, 35, 2896)
        }, {
            'ori_shape': (1, 2, 10, 10),
            'ori_format': 'NCHW',
            'format': 'FRACTAL_Z',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (16, 31), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (16, 31), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, {
            'shape': (-1, 1, -1, -1, 16),
            'ori_shape': (-1, 2, -1, -1),
            'range': ((32, 2**31 - 1), (1, 1), (32, 63), (1024, 4096), (16, 16)),
            'ori_range': ((32, 2**31 - 1), (2, 2), (32, 63), (1024, 4096)),
            'ori_format': 'NCHW',
            'format': 'NC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW',
        'test_conv2d_backprop_input_fuzz_build_upper_limit', {"mode": "keep_rank"}]
    conv2d_backprop_input_generalization(*input_list)

def _test_transdata_dx_transdata_fusion_op():
    """
    transdata + dx + transdata
    dy: nchw -> nc1hwc0
    filter: fz
    dx: nc1hwc0 -> nchw
    """
    dtype = "float16"
    with OpContext("dynamic"):
        with operation.ComputeContext():
            y = {
                'shape': [-1, -1, -1, -1, -1],
                'ori_format': "NCHW",
                'ori_shape': [-1, -1, -1, -1],
                'dtype': dtype,
                'range': [(1, None)] * 4
            }

            var_n = operation.var("batch_n")
            var_dedy_c = operation.var("dedy_c")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
            dy_shape_nchw = (var_n, var_dedy_c, var_dedy_h, var_dedy_w)
            filter_fz = (var_filter_ci1hw, var_filter_co1, 16, 16)

            transdata_in_tensor = tvm.placeholder(dy_shape_nchw, name="transdata_in", dtype=dtype)
            dy_tensor = trans_data_fusion_compute(transdata_in_tensor, {}, "NCHW", "NC1HWC0")
            input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
            filter_tensor = tvm.placeholder(filter_fz, name="filter", dtype=dtype)
            tensor_list = [input_tensor, filter_tensor, transdata_in_tensor]
            _build_dx_transdata_fusion_op(tensor_list, y, (1, 1, 1, 1), dy_tensor, "transdata_dx_transdata_fusion_binary")

def _build_dx_transdata_fusion_op(tensor_list, y, stride, dy_tensor, case_name):
    pads = [-1, -1, -1, -1]
    conv_res = conv2dbp_input_fusion_compute(
        tensor_list[0], tensor_list[1], dy_tensor, y, stride, pads, data_format="NCHW", kernel_name=case_name)
    tran_data_res = trans_data_fusion_compute(conv_res, {"ori_shape": y.get("ori_shape")}, "NC1HWC0", "NCHW")

    tensor_list.append(tran_data_res)
    with tvm.target.cce():
        sch = auto_schedule(tran_data_res)
    config = {
        "name": case_name,
        "tensor_list": tensor_list,
        "build_args": {"constant_realize_extend_in_infer_bound": False}
    }
    build(sch, config)

def _test_dx_transdata_fusion_op():
    """
    dx + transdata
    dy: nc1hwc0
    filter: fz
    dx: nc1hwc0 -> nchw
    """
    dtype = "float16"
    with OpContext("dynamic"):
        with operation.ComputeContext():
            y = {
                'shape': [-1, -1, -1, -1, -1],
                'ori_format': "NCHW",
                'ori_shape': [-1, -1, -1, -1],
                'dtype': dtype,
                'range': [(1, None)] * 4
            }

            var_filter_ci1hw = operation.var("filter_ci1hw")
            var_filter_co1 = operation.var("filter_co1")
            var_n = operation.var("batch_n")
            var_dedy_c1 = operation.var("dedy_c1")
            var_dedy_h = operation.var("dedy_h")
            var_dedy_w = operation.var("dedy_w")
            dy_shape_nc1hwc0 = (var_n, var_dedy_c1, var_dedy_h, var_dedy_w, 16)
            filter_fz = (var_filter_ci1hw, var_filter_co1, 16, 16)

            input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
            dy_tensor = tvm.placeholder(dy_shape_nc1hwc0, name="dedy", dtype=dtype)
            filter_tensor = tvm.placeholder(filter_fz, name="filter", dtype=dtype)

            tensor_list = [input_tensor, filter_tensor, dy_tensor]
            _build_dx_transdata_fusion_op(tensor_list, y, (1, 1, 2, 2), dy_tensor, "dx_transdata_fusion_binary")


if __name__ == '__main__':
    test_conv2d_backprop_input_fuzz_build_lower_limit()
    test_conv2d_backprop_input_fuzz_build_upper_limit()
    _test_dx_transdata_fusion_op()
    _test_transdata_dx_transdata_fusion_op()