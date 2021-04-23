#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2DBackpropInput", "impl.dynamic.conv2d_backprop_input", "conv2d_backprop_input")


def test_conv2d_backprop_input_vadd_drelu_dynamic(test_arg):
    from tbe import tvm
    from te.lang.cce import cce_build_code as build
    from tbe.common.context.op_context import OpContext
    from te.utils.cce import auto_schedule
    import tbe.dsl.base.operation as tbe_operation
    from te import platform as cce_conf
    from impl.dynamic.conv2d_backprop_input import conv2dbp_input_fusion_compute
    from impl.dynamic.relu_grad_v2 import relu_grad_v2_compute
    from impl.dynamic.add_n import add_n_fusion_compute
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
    print("[ conv2d_backprop_input_vadd_drelu_dynamic_case ]")
    case = [{"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
            {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),
             "ori_format": "NCHW", "range":[(1, 1), (16, 16), (2, 2), (2, 2)]},
            {"shape": (16,1,3,3,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (16,16,3,3),
             "ori_format": "NCHW", "range":[(16, 16), (16, 16), (3, 3), (3, 3)]},
            {"shape": (1,1,-1,-1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,16,-1,-1),
             "ori_format": "NCHW", "range":[(1, 1), (16, 16), (4, 4), (4, 4)]},
            [0,0,2,2], [0,0,0,0], "NCHW"]
    input_size = case[0]
    out_backprop = case[1]
    filters = case[2]
    y = case[3]
    strides = case[4]
    pads = case[5]
    dilations = (1, 1, 1, 1)
    groups = 1
    data_format = case[6]
    kernel_name = 'conv2d_backprop_input_vadd_drelu_dynamic_' + cce_conf.cce_conf.get_product_version()
    with OpContext("dynamic") as opc:
        with tbe_operation.ComputeContext():
            conv2dbp_input_info = conv2dbp_input_fusion_compute(input_size, filters, out_backprop, y, strides, pads, 
                                                 dilations, groups, data_format, kernel_name)
            conv2dbp_input_res = conv2dbp_input_info.get("op_res")[0]
            vadd_shape = shape_to_list(conv2dbp_input_res.shape)
            vadd_tensor = tvm.placeholder(vadd_shape, name="vadd_tensor", dtype='float16')
            vadd_info = add_n_fusion_compute([conv2dbp_input_res, vadd_tensor], {}, "", 2)
            vadd_res = vadd_info.get("op_res")[0]
            mask_shape = shape_to_list(conv2dbp_input_res.shape)
            mask = tvm.placeholder(mask_shape, name="mask", dtype='uint1')
            relu_grad_v2_res = relu_grad_v2_compute(vadd_res, mask, {}, "relu_grad_v2")
            with tvm.target.cce():
                sch = auto_schedule(relu_grad_v2_res)
            tensor_list = list(conv2dbp_input_info['op_placeholder']) + [vadd_tensor, mask, relu_grad_v2_res]
            config = {"name": kernel_name,
                      "tensor_list": tensor_list,
                      "build_args": {"constant_realize_extent_in_infer_bound": False}}
        build(sch, config)

ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_vadd_drelu_dynamic)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv2d_backprop_input_vadd_drelu_dynamic)
    exit(0)