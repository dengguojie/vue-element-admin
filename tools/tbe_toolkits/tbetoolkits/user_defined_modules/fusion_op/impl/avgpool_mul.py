import sys
import math

from te import tvm
import te.lang.cce as tbe
from impl.div import div_compute

from impl.util import util_conv2d


def _pad_compute(padding, input_h, input_w, stride, window, dilations=(1,1)):
    """
    Calculate the pad value.
    :param padding: str, SAME or VALID
    :param input_h: int, input h
    :param output_w: int, output w
    :param stride: list, stride attr
    :param window: list, window attr
    :param dilations: list, dilations attr
    :return: pad
    """

    if padding == "SAME":
        He=(window[0] - 1) * dilations[0] + 1
        We=(window[1] - 1) * dilations[1] + 1
        output_h = (input_h + stride[0] - 1) // stride[0]
        output_w = (input_w + stride[1] - 1) // stride[1]
        pad_row = max(0, (output_h - 1) * stride[0] + He - input_h)
        pad_col = max(0, (output_w - 1) * stride[1] + We - input_w)
        pad_top = pad_row // 2
        pad_bottom = pad_row - pad_top
        pad_left = pad_col // 2
        pad_right = pad_col - pad_left
        pad = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        pad = (0, 0, 0, 0)
    return pad

def avgpool_mul(x, filter, assist_matrix, bias, y, ksize, strides,
             padding="VALID", data_format="NHWC",offset_x=0,
             kernel_name="avg_pool"):
    shape_in = list(x.get("ori_shape"))
    shape_w = list(filter.get("ori_shape"))
    if data_format in ("NHWC",):
        input_h = shape_in[1]
        input_w = shape_in[2]
        ksize_h = ksize[1]
        ksize_w = ksize[2]
        window = [ksize[1], ksize[2]]
        stride = [strides[1], strides[2]]
        shape_w[3] = shape_in[3]
        groups = shape_in[3]
    else:
        input_h = shape_in[2]
        input_w = shape_in[3]
        ksize_h = ksize[2]
        ksize_w = ksize[3]
        window = [ksize[2], ksize[3]]
        stride = [strides[2], strides[3]]
        shape_w[1] = shape_in[1]
        groups = shape_in[1]

    pad = list(_pad_compute(padding, input_h, input_w, stride, window))
    bias_tensor = None
    bias_flag = False

    cin_ori = 1
    cout_ori = 1
    enlarge = min(util_conv2d.lcm(util_conv2d.lcm(cin_ori, 16)//cin_ori,
                util_conv2d.lcm(cout_ori, 16)//cout_ori), groups)
    c1_opt = math.ceil(cin_ori*enlarge/16)
    cout1_opt = math.ceil(cout_ori*enlarge/16)
    group_opt = math.ceil(groups/enlarge)
    c1in_ori_align = math.ceil(cin_ori*groups/16)

    in_dtype = x.get("dtype")
    filter_dtype = filter.get("dtype")
    res_dtype = y.get("dtype")
    offset_w_dtype = 'int32'
    optim_dict = {"c0_optim_flg": False, "use_v200_c04_flg": False}
    fusion_para = {"input_memory_type": 0, "output_memory_type": 0,
                    "valid_shape": (), "slice_offset": (),
                    "l1_fusion_type": -1,
                    "fmap_l1_addr_flag": 0,
                    "fmap_l1_valid_size": -1}
    shape_in, shape_w = util_conv2d.conv_layer_cce_para_check(shape_in, shape_w, pad[0:2], pad[2:4],
                                                            stride[0], stride[1], in_dtype, filter_dtype,
                                                            res_dtype, offset_w_dtype, bias_flag,
                                                            kernel_name, 1, 1,
                                                            optim_dict, groups)
    fmap_shape_nc1hwc0, filter_shape_frac_z = util_conv2d.conv_layer_cce_shape_calc(
        shape_in, shape_w, in_dtype, filter_dtype, optim_dict, cout1_opt, c1_opt, group_opt, c1in_ori_align)
    
    with tvm.target.cce():
        data = tvm.placeholder(fmap_shape_nc1hwc0, name="Fmap", dtype=in_dtype)
        weight = tvm.placeholder(filter_shape_frac_z, name="Filter", dtype=in_dtype)
        assist_matrix_input = tvm.placeholder(assist_matrix["shape"], name="assist_matrix", dtype=res_dtype)
        res = tbe.conv(data, weight,
                        para_dict={"bias_tensor": bias_tensor,
                                    "offset_w_tensor": None,
                                    "pad_h": pad[0:2], "pad_w": pad[2:4],
                                    "stride_h": stride[0], "stride_w": stride[1],
                                    "dilate_h": 1, "dilate_w": 1,
                                    "filter_h": ksize_h, "filter_w": ksize_w,
                                    "offset_x": 0, "groups": groups,
                                    "res_dtype": filter_dtype,
                                    "fusion_para": fusion_para,
                                    "kernel_name": kernel_name,
                                    "group": groups,
                                    "enlarge": enlarge,
                                    "c1_opt": c1_opt,
                                    "cout1_opt": cout1_opt,
                                    "group_opt": group_opt,
                                    "a_shape": fmap_shape_nc1hwc0,
                                    "weight_fracz_shape": filter_shape_frac_z,
                                    "weight_ori_shape_nchw": shape_w},
                                    # "padding_mode": padding,
                                    # "pooling_mode": "AVG"},
                        optim_dict=optim_dict)
        out = div_compute(res, assist_matrix_input, None)
        sch = tbe.auto_schedule(out)
        tensor_list = [data, weight, assist_matrix_input, out]

    config = {"print_ir": False,
            "need_build": True,
            "name": kernel_name,
            "tensor_list": tensor_list}
    tbe.build(sch, config)




    

