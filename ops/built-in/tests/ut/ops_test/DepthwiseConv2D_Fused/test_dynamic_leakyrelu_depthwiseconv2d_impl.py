#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("DepthwiseConv2D", "impl.dynamic.depthwise_conv2d", "depthwise_conv2d")

def test_leakyrelu_depthwise_fusion_testcase(test_arg):
    import json
    from tbe import tvm
    import tbe.dsl as tbe
    from tbe.dsl.unify_schedule.unify_auto_schedule import build
    from impl.dynamic.leaky_relu import leaky_relu_compute
    from impl.dynamic.depthwise_conv2d import depthwise_compute
    from impl.util.platform_adapter import operation
    from impl.util.util_cube_dynamic import Conv2dParaProcess
    from tbe.dsl import auto_schedule
    from tbe.common.context import op_context
    from tbe.tvm.target import cce
    dynamic_depthwise_testcase = []
    # fmap, filter, strides, pads, dilation, bias, dim_range, fusion_mode, data_dlow, data_format, dtype

    # without bias
    dynamic_depthwise_testcase.append([[-1, 32, -1, -1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 10), (32, 32), (2, 10), (2, 10)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, -1, -1], (1, 1, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 1), (32, 32), (10, 25), (10, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 16, -1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 5), (32, 32), (10, 10), (10, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, -1, 1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 5), (32, 32), (10, 25), (10, 10)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 16, 16], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 5), (32, 32), (16, 16), (16, 16)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, -1, 16], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 1), (32, 32), (16, 25), (16, 16)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, 16, -1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), None, [(1, 1), (32, 32), (16, 16), (16, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 5, -1, 101], (5, 5, 5, 5), (0, 0, 5, 5), (2, 2, 2, 2), (1, 1, 1, 1), None, [(1, 1), (5, 5), (5, 201), (101, 101)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 5, -1, 101], (5, 5, 5, 5), (0, 0, 5, 5), (2, 2, 2, 2), (1, 1, 1, 1), None, [(1, 1), (5, 5), (5, 300), (101, 101)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 5, -1, 100], (5, 5, 5, 5), (0, 0, 2, 2), (0, 0, 0, 0), (1, 1, 1, 1), None, [(1, 101), (5, 5), (5, 200), (100, 100)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 13, -1], (32, 32, 1, 1), (0, 0, 4, 4), (0, 0, 0, 0), (1, 1, 1, 1), None, [(1, 101), (32, 32), (13, 13), (1, 113)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 31, -1], (32, 32, 4, 4), (0, 0, 5, 5), (1, 2, 1, 2), (1, 1, 1, 1), None, [(1, 101), (32, 32), (13, 13), (1, 113)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 5, -1, 100], (5, 5, 5, 5), (0, 0, 5, 5), (1, 2, 1, 2), (1, 1, 1, 1), None, [(1, 101), (32, 32), (13, 13), (1, 113)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 11, -1], (32, 32, 1, 1), (0, 0, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1), None, [(1, 101), (32, 32), (11, 11), (1, 111)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[8, 32, 11, -1], (9472, 32, 5, 5), (0, 0, 2, 2), (2, 2, 2, 2), (1, 1, 1, 1), None, [(8, 8), (32, 32), (5, 119), (5, 119)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    # with bias
    dynamic_depthwise_testcase.append([[-1, 64, -1, -1], (3, 3, 64, 64), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 10), (64, 64), (20, 40), (20, 40)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, -1, -1], (1, 1, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 1), (32, 32), (10, 25), (10, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 10, -1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 5), (32, 32), (10, 10), (10, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, -1, 10], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 5), (32, 32), (10, 25), (10, 10)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[-1, 32, 16, 16], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 5), (32, 32), (16, 16), (16, 16)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, -1, 16], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 1), (32, 32), (16, 25), (16, 16)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])
    dynamic_depthwise_testcase.append([[1, 32, 16, -1], (3, 3, 32, 32), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1, 960, 1, 1), [(1, 1), (32, 32), (16, 16), (16, 25)],
                                    "leakyrelu_depthwise", None, "NCHW", "float16"])

    def gen_kernel_name(case):
        fmap, filter, strides, pads, dilations, bias, dim_range, fusion_mode, _, _, _ = case

        shape_filter_info = '_'.join([str(i) for i in filter])
        shape_fmap_info = '_'.join([str(i) for i in fmap if i != -1])
        strides_info = '_'.join([str(i) for i in strides])
        pads_info = '_'.join([str(i) for i in pads])
        dilations_info = '_'.join([str(i) for i in dilations])
        bias_info = "_".join([str(i) for i in bias]) if bias else 'none'

        dynamic_dim = ['n', 'c', 'h', 'w']
        dynamic_mode = "dynamic"
        for key, value in enumerate(fmap):
            if key != 1 and value == '-1':
                dynamic_mode += dynamic_dim[key]
        n_range, _, h_range, w_range = dim_range
        n_info = "_".join([str(i) for i in n_range])
        h_info = '_'.join([str(i) for i in h_range])
        w_info = "_".join([str(i) for i in w_range])
        kernel_name_val = "{}_fusionmode_{}_fmap_{}_n_{}_h_{}_w_{}_filter_{}_stride_{}_pad_{}_d_{}_bias_{}".format(
            dynamic_mode, fusion_mode, shape_fmap_info, n_info, h_info, w_info, shape_filter_info, strides_info,pads_info,dilations_info,bias_info
        )
        return kernel_name_val

    def leaky_relu_depthwise_fusion(fmap, filter, bias, offset_w, out, strides, dilations, pads, data_format,
                                    offset_x=0, kernel_name="depthwise_conv2d", config_para=None):
        fmap_tensor = config_para['input_tensor']
        filter_fracz = ((filter['ori_shape'][0] * filter['ori_shape'][1] * filter['ori_shape'][2] + 15)//16,
                        (filter['ori_shape'][3] + 15)//16, 16, 16)
        filter_w = tvm.placeholder(filter_fracz, dtype='float16', name='filter_w', attrs = filter)
        leaky_relu_res = leaky_relu_compute(fmap_tensor, None, kernel_name= kernel_name)
        for k, value in fmap.items():
            fmap_tensor.op.attrs[k] = value
            leaky_relu_res.op.attrs[k] = value
        tensor_list = [fmap_tensor, filter_w]
        if bias:
            bias_tensor = config_para['bias_tensor']
            for k, value in bias.items():
                bias_tensor.op.attrs[k] = value
            tensor_list.append(bias_tensor)
            depthwise_res = depthwise_compute(leaky_relu_res, filter_w, bias_tensor, offset_w, out, strides,
                                            dilations, pads, data_format, offset_x, kernel_name)
            tensor_list.append(depthwise_res)
        else:
            depthwise_res = depthwise_compute(leaky_relu_res, filter_w, None, offset_w, out, strides, dilations,
                                            pads, data_format, offset_x, kernel_name)
            tensor_list.append(depthwise_res)
        with cce():
            sch = auto_schedule([depthwise_res])
        config = {
            "name": kernel_name,
            "tensor_list": tensor_list,
            "build_args": {"constant_realize_extent_in_infer_bound": False}
        }
        build(sch, config)

    def _shape_to_NC1HWC0(shape, data_format, dtype):
        if data_format.upper() == "NCHW":
            n, c, h, w = shape
        else:
            n, h, w, c = shape
        c0 = 16 if dtype.lower() == "float16" else 32
        c1 = (c + c0 -1 )// c0
        return (n, c1, h, w, c0)

    def _gen_trans_data_case(case):
        fm_shape, w_shape, strides, pads, dilations, bias_shape, dim_range, fusion_mode, data_flow, data_format, dtype = case

        data_format = data_format.upper()
        dtype = dtype.lower()
        bias_dtype = "float16" if dtype == "float16" else "int32"
        out_type = "float16" if dtype == "float16" else "int32"

        x = {
            "shape":_shape_to_NC1HWC0(fm_shape, data_format, dtype),
            "format":"NC1HWC0",
            "ori_shape": fm_shape,
            "ori_format": data_format,
            "dtype": dtype,
            "range": dim_range
        }
        weight_h, weight_w, weight_c, weight_n = w_shape
        weight_range = [(weight_h, weight_h), (weight_w, weight_w), (weight_c, weight_c), (weight_n, weight_n)]
        filter = {
            "ori_shape": w_shape,
            "ori_format": "HWCN",
            "dtype": dtype,
            "range": weight_range
        }
        bias = {
            "shape": _shape_to_NC1HWC0(bias_shape, data_format, bias_dtype),
            "format": "NC1HWC0",
            "ori_shape": bias_shape,
            "ori_format": data_format,
            "dtype": bias_dtype
        } if bias_shape else None

        out_shpae = (x['ori_shape'][0], w_shape[2] * w_shape[3], x["ori_shape"][2], x['ori_shape'][3])
        outputs = {
            "ori_shape": out_shpae,
            "dtype": out_type,
            "ori_format": "NCHW"
        }

        paras = {
            "strides": strides,
            "pads": pads,
            "dilations": dilations,
            "inputs": x,
            "weights": filter,
            "bias": bias,
            "outputs": outputs,
            "data_format": data_format,
            "groups": x['ori_shape'][1],
            "optim_dict": {"c0_optim_flg": False},
            "kernel_name": "leakyrelu_depthwiseconv2d"
        }
        conv_para = Conv2dParaProcess(paras)
        cp = conv_para.config_paras()
        offset_w = None
        offset_x = 0
        return {
            "params": [x, filter, bias, offset_w, strides, dilations, pads,
                    data_format, offset_x, dim_range, outputs],
            "config_para": cp
        }

    def test_leakyrelu_depthwise_fusion(*case):
        with op_context.OpContext("dynamic"):
            with operation.ComputeContext():
                res = _gen_trans_data_case(*case)
                x, weights, bias, offset_w, strides, dilations, pads, data_format, offset_x, dim_range, outputs = res['params']

                kernel_name_val = gen_kernel_name(*case)
                leaky_relu_depthwise_fusion(x, weights, bias, offset_w, outputs, strides, dilations, pads,
                                                        data_format, offset_x, kernel_name_val, res['config_para'])
                print("[info] finish compile kernel_name:", kernel_name_val)

    def run_testcase():
        for case in dynamic_depthwise_testcase:
            test_leakyrelu_depthwise_fusion(case)
    run_testcase()
    print("start run leakyrelu_depthwiseconv2d fusion")

ut_case.add_cust_test_func(test_func=test_leakyrelu_depthwise_fusion_testcase)

if __name__ == '__main__':
    # ut_case.add_cust_test_func(test_func=test_leakyrelu_depthwise_fusion_testcase)
    ut_case.run("Ascend310")
    ut_case.run("Ascend910A")
    exit(0)
