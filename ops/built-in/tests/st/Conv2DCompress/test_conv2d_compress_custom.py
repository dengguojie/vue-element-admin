#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
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
    v200_version = ("Ascend310P", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
    if te.platform.cce_conf.get_soc_spec("SHORT_SOC_VERSION") in v200_version:
        scale_dtype = "uint64"
    deq_scale = tvm.placeholder((1, input_list[1]["ori_shape"][0]//16, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [input_list[1]["ori_shape"][0]]})
    input_params = [x_, filter_, compress_index, None, None, None, *input_list[6:]]
    with tbe.common.context.op_context.OpContext():
        with tbe.dsl.base.operation.compute():
            conv_out = conv2dcompress_compute(*input_params)
            res = ascend_dequant_compute(conv_out, deq_scale, None)
            tensor_list = [x_, filter_, deq_scale, compress_index, res]
            with tvm.target.cce():
                sch = auto_schedule(res)
            config = {
                "name": "conv2d_compress_dequant",
                "tensor_list": tensor_list,
                "build_args": {"constant_realize_extent_in_infer_bound": False}
            }
            te.lang.cce.cce_build_code(sch, config)


def run_v300_conv2d_compress_single_op():
    Ci0_dict = {
        "float32": 8,
        "float16": 16,
        "int8": 32,
        "bfloat16": 16
    }
    bias_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "float32",
        "int8": "int32"
    }
    res_dtype_dict = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "int8": "int32",
        "hf32": "float32"
    }

    def get_input_dict(conv_type, shape_in, shape_w, bias_flag):
        Ni, Ci, Hi, Wi = shape_in
        Cout, Ci, Hk, Wk = shape_w # [in_channels, channel_multiplier, filter_height, filter_width]

        Ci0 = Ci0_dict[conv_type]
        Ci1 = (Ci + Ci0 - 1) // Ci0
        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = [Ci1*Hk*Wk, (Cout + 15) // 16, 16, 32]

        inputs = {
            "ori_shape": shape_in,
            "ori_format": "NCHW",
            "shape": shape_in_5HD,
            "format": "NC1HWC0",
            "dtype": conv_type,
            "is_first_layer": False
        }
        weights = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": shape_w_fracz,
            "format": "FRACTAL_Z",
            "dtype": conv_type,
        }
        bias = {
            "ori_shape": (Cout,),
            "dtype": bias_dtype_dict[conv_type],
            "shape": (Cout,),
            "format": "ND",
            "ori_format": "ND",
        } if bias_flag else None
        outputs = {
            "ori_shape": [],
            "ori_format": "NCHW",
            "dtype": res_dtype_dict[conv_type],
            "shape": [],
            "format": "NC1HWC0",
        }

        return inputs, weights, bias, outputs


    def v300_conv2d_compress_single_op(casename, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag):
        from impl.conv2d_compress import conv2dcompress
        from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
        inputs, weights, bias, outputs = get_input_dict(conv_type, shape_in, shape_w, bias_flag)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        compress_index_shape = [-1, -1, -1, -1]
        compress_index = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": compress_index_shape,
            "format": "FRACTAL_Z",
            "dtype": "int8",
        }
        conv2dcompress(inputs, weights, compress_index, bias, None, outputs, strides, pads,
            dilations, groups=groups, data_format='NCHW', alg=WEIGHT_SPARSE_4_2, kernel_name=casename)

    def v300_conv2d_compress_compute_op(casename, conv_type, shape_in, shape_w, pads, strides,
                                        groups, bias_flag):
        import tvm
        from tbe.dsl import auto_schedule
        from impl.conv2d_compress import conv2dcompress_compute
        from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
        inputs, weights, bias, outputs = get_input_dict(conv_type, shape_in, shape_w, bias_flag)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        with tvm.target.cce():
            fmap = tvm.placeholder(inputs.get("shape"), name='fmap', dtype=conv_type)
            weight = tvm.placeholder(weights.get("shape"), name='weight', dtype=conv_type,
                                     attrs={'ori_shape': shape_w, 'ori_format': "NCHW"})
            compress_index_shape = weights.get("shape")[:-1] + [weights.get("shape")[-1] // 4]

            compress_index = tvm.placeholder(compress_index_shape,
                                             name='weight_index',
                                             dtype=conv_type,
                                             attrs={
                                                 'ori_shape': shape_w,
                                                 'ori_format': "NCHW"
                                             })
            bias = tvm.placeholder(bias.get("shape"), name='bias',
                                   dtype=bias.get("dtype")) if bias_flag else None
            out = conv2dcompress_compute(fmap, weight, compress_index, bias, None, outputs,
                                         strides, pads, dilations, groups=groups,
                                         data_format='NCHW',
                                         alg=WEIGHT_SPARSE_4_2, kernel_name="conv2dcompress",
                                         options=None)
            sch = auto_schedule(out)


    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.context import op_context
    v300_cases = [
        ("conv2d_add", "conv2d_add", "int8", (1, 3, 17, 17), (64, 3, 7, 7), (3, 3, 3, 3), (1, 1), 1, 1, False, 0, 0, 0)
    ]
    set_current_compile_soc_info("Ascend310B1")
    with op_context.OpContext():
        for case in v300_cases:
            print("=" * 150)
            print("case {}".format(case))
            casename, dataflow, conv_type, shape_in, shape_w, pads, strides, \
                dilation, groups, bias_flag, quant_scale, quant_offset, relu_param = case
            # v300_conv2d_compress_single_op(casename, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag)
            v300_conv2d_compress_compute_op(casename, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag)


if __name__ == '__main__':
    run_v300_conv2d_compress_single_op()
    soc_version = cce_conf.get_soc_spec("FULL_SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_conv2d_compress_no_bias()
    test_conv2d_compress_fused()
    cce_conf.te_set_version(soc_version)
