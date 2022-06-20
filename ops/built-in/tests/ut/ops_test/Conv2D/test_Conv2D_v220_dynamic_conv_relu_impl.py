#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_v220_dynamic(test_arg):

    import tbe
    import json
    from impl.util.platform_adapter import operation
    from impl.dynamic.conv2d import conv2d_fusion_compute
    from impl.dynamic.relu import relu_compute
    from impl.dynamic.ascend_dequant import ascend_dequant_compute
    from tbe.common.context import op_context
    from te import platform as cceconf
    from te import tvm
    from te.utils.cce import auto_schedule
    from tbe.common.platform.platform_info import get_soc_spec
    from tbe.dsl.unify_schedule.build import build
    import pdb

    v220_dynamic_case = [
        #           dataflow,
        #           conv_type,
        #           shape_in,
        #           shape_w,
        #           dim_range, # [batch_range, _, h_range, w_range]
        #           pads,
        #           strides,
        #           groups,
        #           bias_flag,
        #           (relu_mode), "prelu/relu/None", prelu only allows fp16 and fp32 dtype as input.
        #           (quant_scale),
        #           (quant_offset),
        #           (sqrt_mode) # only for dequant
        ("conv2d_relu", "float16", (1, 64, -1, -1), (64, 64, 3, 3), [(1, 1), (64, 64), (50, 60), (50, 60)], (1, 1, 1, 1), (1,1), 1, False),
    ]

    def conv_v220_dynamic_fusion_case(dataflow,
                                      conv_type,
                                      in_nd2nz_flag,
                                      out_nz2nd_flag,
                                      shape_in,
                                      shape_w,
                                      dim_range,
                                      pads,
                                      strides,
                                      groups,
                                      bias_flag,
                                      relu_mode=None,
                                      quant_scale=0,
                                      quant_offset=0,
                                      sqrt_mode=False,
                                      cout_real=0):
        if in_nd2nz_flag:
            Ni, Hi, Wi, Ci = shape_in
            h_index, w_index = 1, 2
        else:
            Ni, Ci, Hi, Wi = shape_in
            h_index, w_index = 2, 3

        Co, w_Ci, Hk, Wk = shape_w

        range_in = dim_range
        range_w = [(Co, Co), (w_Ci, w_Ci), (Hk, Hk), (Wk, Wk)]

        Ci0_dict = {
            "float32": 8,
            "float16": 16,
            "int8": 32,
            "bfloat16": 16
        }
        Ci0 = Ci0_dict[conv_type]
        Ci1 = (Ci + Ci0 - 1) // Ci0

        Co0 = 16
        Co1 = (Co + Co0 - 1) // Co0

        if Ni == -1:
            Ni = operation.var("batch_n", range_in[0])
            operation.add_exclude_bound_var(Ni)
        if Hi == -1:
            Hi = operation.var("fmap_h", range_in[h_index])
            operation.add_exclude_bound_var(Hi)
        if Wi == -1:
            Wi = operation.var("fmap_w", range_in[w_index])
            operation.add_exclude_bound_var(Wi)
        
        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = (Hk * Wk * Ci1, Co1, Co0, Ci0)
        shape_in_expr = (Ni, Hi, Wi, Ci) if in_nd2nz_flag else (Ni, Ci, Hi, Wi)

        shape_scale = (1, Co1, 1, 1, 16)
        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]

        bias_dtype_dict = {
            "float32": "float32",
            "float16": "float32",
            "bfloat16": "float32",
            "int8": "int32"
        }
        bias_dtype = bias_dtype_dict[conv_type]

        fmap = tvm.placeholder(shape_in_5HD,
                               name='fmap',
                               dtype=conv_type,
                               attrs={
                                   'ori_shape': shape_in,
                                   'ori_format': 'NCHW',
                                   'range': range_in,
                                   'format': 'NC1HWC0'
                               })
        
        data_format = 'NCHW'

        weight = tvm.placeholder(shape_w_fracz,
                                 name='weight',
                                 dtype=conv_type,
                                 attrs={
                                     'ori_shape': shape_w,
                                     'ori_format': 'NCHW',
                                     'range': range_w,
                                     'format': "FRACTAL_Z"
                                 })
        bias = tvm.placeholder(
            (Co1 * Co0, ), name='bias', dtype=bias_dtype) if bias_flag else None
        out_shape = [Ni, Co, Hi, Wi]
        res_dtype_dict = {
            "float32": "float32",
            "float16": "float16",
            "bfloat16": "bfloat16",
            "int8": "int32"
        }
        outputs = {
            "ori_shape": out_shape,
            "ori_format": "NCHW",
            "dtype": res_dtype_dict[conv_type]
        }

        conv_res = conv2d_fusion_compute(fmap,
                                         weight,
                                         bias,
                                         None,
                                         outputs,
                                         strides,
                                         pads,
                                         dilations,
                                         data_format=data_format)
        vdeq = tvm.placeholder(shape_scale,
                               name='vreq_reg',
                               dtype='uint64',
                               attrs={'ori_shape': [Co1 * Co0]})

        relu_flag = True if relu_mode == "relu" else False

        if dataflow == "conv2d_relu":
            out = relu_compute(conv_res, None)
        
        if dataflow == "conv2d_relu":
            tensor_list = [fmap, weight, out]
        
        if bias_flag:
            tensor_list.insert(2, bias)
        
        with tvm.target.cce():
            sch = auto_schedule(out)
        
        config = {
            "name": "conv2d_fusion",
            "tensor_list": tensor_list,
            "build_args": {
                "constant_realize_extent_in_infer_bound": False
            }
        }
        build(sch, config)
    
    def run_testcase(config_dict):
        for i in config_dict:
            # try:
            print("=" * 150)
            print("case {}".format(i))
            print()

            in_nd2nz_flag = False
            out_nz2nd_flag = False
            cout_real = 0

            if i[0].startswith("nd2nz_"):
                in_nd2nz_flag = True
                dataflow = i[0][6:]
            elif i[0].endswith("_nz2nd"):
                out_nz2nd_flag = True
                dataflow = i[0][:-6]
                cout_real = i[5]
                i = i[:5] + i[6:]
            else:
                dataflow = i[0]
            
            if dataflow == "conv2d_single_op":
                _, conv_type, shape_in, shape_w, dim_range, pads, strides, groups, bias_flag = i
                with op_context.OpContext("dynamic"):
                    with tbe.dsl.base.operation.dynamic():
                        with tbe.dsl.base.operation.compute():
                            conv_v220_dynamic_single_op_case(conv_type, in_nd2nz_flag,
                                                             out_nz2nd_flag, shape_in, shape_w,
                                                             dim_range, pads, strides, groups,
                                                             bias_flag)
            else:
                _, conv_type, shape_in, shape_w, dim_range, pads, strides, groups, bias_flag = i
                with op_context.OpContext("dynamic"):
                    with tbe.dsl.base.operation.dynamic():
                        with tbe.dsl.base.operation.compute():
                            conv_v220_dynamic_fusion_case(dataflow,
                                                          conv_type,
                                                          in_nd2nz_flag,
                                                          out_nz2nd_flag,
                                                          shape_in,
                                                          shape_w,
                                                          dim_range,
                                                          pads,
                                                          strides,
                                                          groups,
                                                          bias_flag,
                                                          cout_real=cout_real)

    def test_dynamic_conv2d_split_info():
        from impl.dynamic.conv2d import get_op_support_info

        input_list = [
            {
                'ori_shape': (-1, 64, 32, 32),
                'ori_format': 'NCHW',
                'format': 'NC1HWC0',
                'dtype': 'float16'
            }, {
                'format': 'FRACTAL_Z',
                'dtype': 'float16'
            }, None, None, {
                'format': 'NC1HWC0',
                'dtype': 'float16'
            }, (1, 1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), 1, 'NCHW']
        get_op_support_info(*input_list)

    cceconf.te_set_version('Ascend910B2')
    run_testcase(v220_dynamic_case)
    test_dynamic_conv2d_split_info()


print("adding Conv2D v220 dynamic ut testcases")
ut_case.add_cust_test_func("Ascend910B2", test_func=test_conv2d_v220_dynamic)
ut_case.run(['Ascend910B2'])