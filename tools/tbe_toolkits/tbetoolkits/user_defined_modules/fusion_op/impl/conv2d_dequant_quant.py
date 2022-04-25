# -*- coding:utf-8 -*-

import sys

import te
import tbe
from te import tvm

from topi import generic
from impl.conv2d import conv2d_compute
from impl.leaky_relu import leaky_relu_compute
#from te.lang.cce import AutoScheduleOp
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_quant import ascend_quant_compute
from te.platform.cce_conf import get_soc_spec



def icd(num_a, num_b):
    """
    upper division
    """
    return (num_a + num_b - 1) // num_b
def lcm(wout, factor):
    """
    get least common multiple of wout and factor
    """
    tmp = wout*factor
    while wout % factor != 0:
        wout, factor = factor, (wout % factor)
    return tmp // factor

def conv2d_dequant_quant(inputs, weights, bias, deg_scle,offset_w, outputs, strides, pads, dilations=[1, 1, 1, 1],
           groups=1, data_format='NHWC', offset_x=0, kernel_name="conv2d_dequant_quant"):
    print("mmmmmmmmmmmmmmmmm",inputs, weights, bias, offset_w, outputs, strides, pads, dilations,groups,data_format,offset_x,kernel_name)

#    {'shape': (4, 6, 4, 517, 32), 'ori_shape': (4, 192, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'} {'shape': (384, 6, 1, 4, 32), 'ori_shape': (384, 192, 1, 4), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'} {'shape': (384,), 'ori_shape': (384,), 'range': None, 'dtype': 'int8', 'format': None, 'ori_format': None} None {'shape': (4, 12, 4, 517, 32), 'ori_shape': (4, 384, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'} (0, 0, 1, 1) (0, 0, 1, 2) (1, 1, 1, 1) 1 NCHW 0 static_op_conv2d_dequant_quant_0
    #inputs = {'shape': (4, 6, 4, 517, 32), 'ori_shape': (4, 192, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
    #weights = {'shape': (384, 6, 1, 4, 32), 'ori_shape': (384, 192, 1, 4), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
    #bias = {'shape': (384,), 'ori_shape': (384,), 'range': None, 'dtype': 'int8', 'format': None, 'ori_format': None}
    #offset_w = None
    #outputs = {'shape': (4, 12, 4, 517, 32), 'ori_shape': (4, 384, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
    #strides = (0, 0, 1, 1)
    #pads = (0, 0, 1, 2) 
    #dilations =  (1, 1, 1, 1)
    #groups = 1 
    #data_format = "NCHW"
    #offset_x = 0
    #kernel_name = "static_op_conv2d_dequant_quant_0"

    soc_version = get_soc_spec("SOC_VERSION")
    block_size_n = 16
    block_size_k = 32  # 16 if dtype='fp16' else 32
    C0 = 32  # 16 if dtype='fp16' else 32

    N, C, H, W = inputs['ori_shape']
    c_per_group= weights['ori_shape'][1] // groups
    Co,Cin_k,kh,kw= weights['ori_shape']
    weights['ori_shape'] = (Co,c_per_group,kh,kw)

    filter= weights['ori_shape']
    C1 = (C + C0 - 1) // C0
    shape_in = (N, C1, H, W, C0)
    #fp16
    Co1 = (Co + block_size_n - 1) // block_size_n
    #int
    #Co1 = (Co + block_size_k - 1) // block_size_k * block_size_k
    #Co1 = (Co1 + block_size_n - 1) // block_size_n
    #in_channel_weight = ((C + block_size_k - 1) // block_size_k) * block_size_k
    #shape_w = ((in_channel_weight * kh * kw + block_size_k - 1) // block_size_k,
    #        Co1, block_size_n, block_size_k)
    cin_per_group = weights["ori_shape"][1]
    cout_per_group = weights["ori_shape"][0]//groups

    enlarge = min(lcm(lcm(cin_per_group, block_size_k)//cin_per_group, lcm(cout_per_group, block_size_n)//cout_per_group),
                  groups)
    cin1_per_group_opt = icd(cin_per_group*enlarge, block_size_k)
    cout1_per_group_opt = icd(cout_per_group*enlarge, block_size_n)
    group_opt = icd(groups, enlarge)

    shape_w = (group_opt*weights["ori_shape"][2]*weights["ori_shape"][3]*cin1_per_group_opt, cout1_per_group_opt, block_size_n, block_size_k)

    quant_dict = {'scale': 10.32, 'sqrt_mode': False, 'offset': 0.5, 'round_mode':'Round'}
    shape_deq_scale  = (1, weights["shape"][0], 1, 1, 16) 
    shape_deq_scale  = (1,Co1, 1, 1, 16) 
    with tvm.target.cce():
        # conv2d    
        fm = tvm.placeholder(shape_in, name='fm', dtype='int8', attrs={'ori_format': 'NCHW'})
        filter_w = tvm.placeholder(shape_w, name='filter_w', dtype='int8',
                            attrs={'ori_shape': filter, 'ori_format': 'NCHW'})
        if bias:
            bias_tensor = tvm.placeholder((Co1*16,), name='bias', dtype='int32')
        else:
            bias_tensor = None
        conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations,groups, data_format, offset_x, kernel_name)
        deq_reg = tvm.placeholder(shape_deq_scale, dtype="float16", name="deq_reg",attrs={'ori_shape': (weights.get("ori_shape")[0],)})
        if soc_version in "Ascend710,Ascend610,Hi3796CV300CS":
            deq_reg = tvm.placeholder(shape_deq_scale, dtype="uint64", name="deq_reg",attrs={'ori_shape': [weights.get("ori_shape")[0] ]})
        dequant_res = ascend_dequant_compute(conv_res, deq_reg, None, sqrt_mode=False, relu_flag=False)
        out = ascend_quant_compute(dequant_res, None, quant_dict["scale"], quant_dict["offset"], quant_dict["sqrt_mode"])

        sch = generic.auto_schedule(out)
        tensor_list = [fm, filter_w, deq_reg, out]
        
        if bias:
            tensor_list = [fm, filter_w, bias_tensor, deq_reg, out]

    config = {"print_ir": False,
            "need_build": True,
            "name": kernel_name,
            "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)

#if __name__ == "__main__":
#    inputs = {'shape': (4, 6, 4, 517, 32), 'ori_shape': (4, 192, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
#    weights = {'shape': (384, 6, 1, 4, 32), 'ori_shape': (384, 192, 1, 4), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
#    bias = {'shape': (384,), 'ori_shape': (384,), 'range': None, 'dtype': 'int8', 'format': None, 'ori_format': None}
#    offset_w = None
#    outputs = {'shape': (4, 12, 4, 517, 32), 'ori_shape': (4, 384, 4, 517), 'range': None, 'dtype': 'int8', 'format': 'NC1HWC0', 'ori_format': 'NCHW'}
#    strides = (0, 0, 1, 1)
#    pads = (0, 0, 1, 2)
#    dilations =  (1, 1, 1, 1)
#    groups = 1
#    data_format = "NCHW"
#    offset_x = 0
#    kernel_name = "static_op_conv2d_dequant_quant_0"
#    conv2d_dequant_quant(inputs, weights, bias, offset_w, outputs, strides, pads, dilations,groups,data_format,offset_x,kernel_name)
#    print("every thing is  ok ")
