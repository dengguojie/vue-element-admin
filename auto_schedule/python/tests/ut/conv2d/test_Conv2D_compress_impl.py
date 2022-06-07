#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("conv_schedule", "conv_schedule.test_static_conv_schedule_impl")


def run_v300_conv2d_compress(test_arg):
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
        compress_index_shape = weights.get("shape")[:-1] + [weights.get("shape")[:-1] // 4]
        compress_index = {
            "ori_shape": shape_w,
            "ori_format": "NCHW",
            "shape": compress_index_shape,
            "format": "FRACTAL_Z",
            "dtype": "int8",
        }
        conv2dcompress(inputs, weights, compress_index, bias, None, outputs, strides, pads,
            dilations, groups=groups, data_format='NCHW', alg=WEIGHT_SPARSE_4_2, kernel_name=casename)

    def v300_conv2d_compress_compute_op(casename, conv_type, shape_in, shape_w, pads, strides, groups, bias_flag):
        import tvm
        from tbe.dsl import auto_schedule
        from impl.conv2d_compress import conv2dcompress_compute
        from tbe.common.utils.op_util.op_util_conv2d import WEIGHT_SPARSE_4_2
        inputs, weights, bias, outputs = get_input_dict(conv_type, shape_in, shape_w, bias_flag)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]
        with tvm.target.cce():
            fmap = tvm.placeholder(inputs.get("shape"), name='fmap', dtype=conv_type)
            weight = tvm.placeholder(weights.get("shape"), name='weight', dtype=conv_type, attrs={'ori_shape': shape_w, 'ori_format': "NCHW"})
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
                               strides, pads, dilations, groups=groups, data_format='NCHW',
                               alg=WEIGHT_SPARSE_4_2, kernel_name="conv2dcompress", options=None)
            sch = auto_schedule(out)


    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.context import op_context
    v300_cases = [
        ("conv2d", "conv2d", "int8", (1, 3, 17, 17), (64, 3, 7, 7), (3, 3, 3, 3), (1, 1), 1, 1, False, 0, 0, 0)
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


ut_case.add_cust_test_func(test_func=run_v300_conv2d_compress)

if __name__ == "__main__":
    ut_case.add_cust_test_func(test_func=run_v300_conv2d_compress)
    ut_case.run("Ascend310B1")
    exit(0)
