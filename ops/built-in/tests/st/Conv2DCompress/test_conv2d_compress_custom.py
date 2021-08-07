#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
import topi
from te import tvm
from te import platform as cce_conf
from impl.conv2d_compress import conv2dcompress
from impl.conv2d_compress import conv2dcompress_compute
from impl.ascend_dequant import ascend_dequant_compute


def test_conv2d_compress_no_bias():
    '''
    for conv2d_compress single op
    '''
    input_list = [{
        'ori_shape': (128, 3, 64, 64),
        'shape': (128, 1, 64, 64, 32),
        'ori_format': 'NCHW',
        'format': 'NC1HWC0',
        'dtype': 'int8'
    }, {
        'ori_shape': (64, 3, 4, 4),
        'shape': (16, 4, 16, 32),
        'ori_format': 'NCHW',
        'format': 'FRACTAL_Z',
        'dtype': 'int8'
    }, {
        'ori_shape': (64, 3, 4, 4),
        'ori_format': 'ND',
        'dtype': 'int8'
    }, None, None, {
        'format': 'NC1HWC0',
        'dtype': 'int32'
    }, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
    with tbe.common.context.op_context.OpContext():
        conv2dcompress(*input_list)

def test_conv2d_compress_fused():
    '''
    for conv2d_compress fusion op
    '''
    input_list = [{
        'ori_shape': (128, 3, 64, 64),
        'shape': (128, 1, 64, 64, 32),
        'ori_format': 'NCHW',
        'format': 'NC1HWC0',
        'dtype': 'int8'
    }, {
        'ori_shape': (64, 3, 4, 4),
        'shape': (16, 4, 16, 32),
        'ori_format': 'NCHW',
        'format': 'FRACTAL_Z',
        'dtype': 'int8'
    }, {
        'ori_shape': (64, 3, 4, 4),
        'ori_format': 'ND',
        'dtype': 'int8'
    }, None, None, {
        'format': 'NC1HWC0',
        'dtype': 'int32'
    }, (1, 1, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']

    x_ = tvm.placeholder(input_list[0]["shape"], name="x", dtype=input_list[0]["dtype"], attrs=input_list[0])
    filter_ = tvm.placeholder(input_list[1]["shape"], name="filter", dtype=input_list[1]["dtype"], attrs=input_list[1])
    compress_index_shape = tvm.var("compress_index_shape", dtype="int32")
    compress_index = tvm.placeholder([compress_index_shape], name="compress_index", dtype="int8")

    scale_dtype = "float16"
    v200_version = ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
    if te.platform.cce_conf.get_soc_spec("SOC_VERSION") in v200_version:
        scale_dtype = "uint64"
    deq_scale = tvm.placeholder((1, input_list[1]["ori_shape"][0]//16, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [input_list[1]["ori_shape"][0]]})
    input_params = [x_, filter_, compress_index, None, None, None, *input_list[6:]]
    with tbe.common.context.op_context.OpContext():
        with tbe.dsl.base.operation.compute():
            conv_out = conv2dcompress_compute(*input_params)
            res = ascend_dequant_compute(conv_out, deq_scale, None)
            tensor_list = [x_, filter_, deq_scale, compress_index, res]
            with tvm.target.cce():
                sch = topi.generic.auto_schedule(res)
            config = {
                "name": "conv2d_compress_dequant",
                "tensor_list": tensor_list,
                "build_args": {"constant_realize_extent_in_infer_bound": False}
            }
            te.lang.cce.cce_build_code(sch, config)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_conv2d_compress_no_bias()
    test_conv2d_compress_fused()
    cce_conf.te_set_version(soc_version)
