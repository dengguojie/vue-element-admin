# -*- coding:utf-8 -*-

import sys

import te
from tbe import tvm
import tbe
from tbe.common.register import register_operator
import tbe.dsl.base as tbe_base
from tbe.dsl.base import operation
from tbe.dsl import build
from impl.dynamic.conv2d import conv2d_fusion_compute as conv
from impl.dynamic.sigmoid import sigmoid_compute as sigmoid


def icd(num_a, num_b):
    """
    upper division
    """
    return (num_a + num_b - 1) // num_b


def lcm(wout, factor):
    """
    get least common multiple of wout and factor
    """
    tmp = wout * factor
    while wout % factor != 0:
        wout, factor = factor, (wout % factor)
    return tmp // factor

@register_operator("Conv2D")
def conv2d_sigmoid(inputs,weights,bias,offset_w,outputs,strides,pads,dilations,groups, data_format, offset_x,kernel_name='conv2d_sigmoid_dynamic'):
    add_outputs={"dtype": "float16"}
    with tbe.dsl.base.operation.compute():

        fm_shape = list(inputs.get("shape"))
        in_range_nchw = inputs.get("range")
        if inputs.get("ori_format") == "NCHW" and inputs.get("format") == "NC1HWC0":
            n_dim = 0
            h_dim = 2
            w_dim = 3
            dynamic_flag = -1
            unknown_flag = -2
        else:
            print("input format is not NCHW,please change to NCHW")

        if fm_shape[n_dim] == dynamic_flag:
            fm_shape[n_dim] = operation.var("batch_n", in_range_nchw[n_dim])
            # in_shape_nc1hwc0[N_DIM] = tbe_base.var("batch_n", in_range[N_DIM])
            operation.add_exclude_bound_var(fm_shape[n_dim])
        if fm_shape[h_dim] == dynamic_flag:
            fm_shape[h_dim] = operation.var("fmap_h", in_range_nchw[h_dim])
            operation.add_exclude_bound_var(fm_shape[h_dim])
        if fm_shape[w_dim] == dynamic_flag:
            fm_shape[w_dim] = operation.var("fmap_w", in_range_nchw[w_dim])
            operation.add_exclude_bound_var(fm_shape[w_dim])

        input_tensor = tvm.placeholder(
            fm_shape, name="fmap", dtype=inputs.get("dtype"),
            attrs={'shape':  tuple(fm_shape),
                   "ori_shape": inputs.get("ori_shape"),
                   "format": inputs.get("format"),
                   "ori_format": inputs.get("ori_format"),
                   "range": inputs.get("range")})
        print("########################", input_tensor.shape)


        Cout, Cin, filter_h, filter_w, C0 = weights.get("shape")

        #转成fraZ格式输入
        if C0 == 16:
            Cout = (Cout + 15) // 16
            # 非量化场景 C0=16
            block_size_n = block_size_k = C0

            # 量化场景C0=32，block_size_n = 16, block_size_k=32
            weights_fraz = (
            Cin * filter_h * filter_w, Cout, block_size_n, block_size_k)
            if groups > 1:
                cin_per_group = weights["ori_shape"][1] // groups
                cout_per_group = weights["ori_shape"][0] // groups

                enlarge = min(
                    lcm(lcm(cin_per_group, block_size_k) // cin_per_group,
                        lcm(cout_per_group, block_size_n) // cout_per_group),
                    groups)
                cin1_per_group_opt = icd(cin_per_group * enlarge, block_size_k)
                cout1_per_group_opt = icd(cout_per_group * enlarge,
                                          block_size_n)
                group_opt = icd(groups, enlarge)

                weights_fraz = (
                group_opt * weights["ori_shape"][2] * weights["ori_shape"][
                    3] * cin1_per_group_opt, cout1_per_group_opt, block_size_n,
                block_size_k)

        else:
            # group > 1和C0！=4的场景
            print("[ERROR] tbetoolkit the input and weight format of case can't"
                  " handle the situation of group >1 and C0=4")

        #weights_fraz = (Cin*filter_h*filter_w, Cout, block_size_n, block_size_k)
        print("$$$$$$$$$$$$", weights_fraz)
        weight_tensor = tvm.placeholder(
            weights_fraz, name="weight_map", dtype=weights.get("dtype"),
            attrs={'shape':  weights_fraz,
                   "ori_shape": weights.get("ori_shape"),
                   "format": "FRACTAL_Z",
                   "ori_format": weights.get("ori_format")})

        bias_tensor = None
        if bias:
            bias_tensor = tvm.placeholder(
                bias.get("ori_shape"),name="bias", dtype=bias.get("dtype"),
                attrs={'shape':  bias.get('shape'),
                       "ori_shape": bias.get("ori_shape"),
                       "format": "ND",
                       "ori_format": "ND"})

        conv_res = conv(input_tensor,
                        weight_tensor,
                        bias_tensor,
                        offset_w,
                        outputs,
                        strides,
                        pads,
                        dilations,
                        groups=groups,
                        data_format=data_format,
                        offset_x=offset_x,
                        kernel_name="conv2d")

        print("AAAAAAAAAAAAAAAAAAAAAAAAAA", conv_res)

        # Tensor(shape=[batch_n, 4, (16*wo), 16], op.name=bias_add_vector_cc_1) <tvm.tensor.TensorSlice object at 0x7f164159a1d0>
        # add_res = add(conv_res['op_res'][0], add_outputs, "add")
        sigmoid_res = sigmoid(conv_res, "sigmoid")
        print("ABABABABBBBBBBBBBBBBBBBBBBBBBBBBBB")
        print(conv_res)
        with tvm.target.cce():
            sch = te.utils.cce.auto_schedule(sigmoid_res)
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
        tensor_list = [input_tensor, weight_tensor, sigmoid_res]
        if bias:
            tensor_list = [input_tensor, weight_tensor, bias_tensor, sigmoid_res]
        config = {"name": kernel_name,
            "tensor_list": tensor_list,
            "build_args": {"constant_realize_extent_in_infer_bound": False}}

    build(sch, config)


