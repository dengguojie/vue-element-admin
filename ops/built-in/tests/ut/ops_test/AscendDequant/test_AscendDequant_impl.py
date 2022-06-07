#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm

from op_test_frame.ut import OpUT
from impl.ascend_dequant import _matmul_compute
from impl.ascend_dequant import ascend_dequant_compute
from impl.ascend_dequant import _vector_dequant_v100
from impl.ascend_dequant import _vector_dequant_v200
from impl.ascend_dequant import _scalar_dequant_v100
from impl.ascend_dequant import _scalar_dequant_v200
from impl.ascend_dequant import _conv_dequant_v200_int4
from impl.ascend_quant_util import get_scale_indices
from impl.ascend_quant_util import get_depthwise_conv2d_tensor_info
from impl.ascend_quant_util import get_conv_flag
from impl.ascend_quant_util import is_conv3d_fuse
from impl.ascend_quant_util import is_support_a100
from impl.ascend_quant_util import is_lhisi_version
from impl.ascend_quant_util import is_support_v200
from impl import ascend_quant_util as util
from te import platform as cce_conf

ut_case = OpUT("AscendDequant", None, None)

case1 = {"params": [{"shape": (1,1,1,1,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (1,2,4,4,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,2,4,4,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,2,4,4,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (1,4,4,16,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": (2,1,1,16,16), "dtype": "int32", "format": "FRACTAL_NZ", "ori_shape": (2,4,4),"ori_format": "NC1HWC0"},
                    {"shape": (1,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,),"ori_format": "ND"},
                    {"shape": (2,1,1,16,16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_4",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (2,1,1,1,16,16), "dtype": "int32", "format": "NDC1HWC0", "ori_shape": (2,1,4,4),"ori_format": "NDC1HWC0"},
                    {"shape": (1,1,1,1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,),"ori_format": "NDHWC"},
                    {"shape": (2,1,1,1,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,1,4,4),"ori_format": "NDC1HWC0"}],
         "case_name": "ascend_dequant_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (2,1,1,1,16,16), "dtype": "int32", "format": "NDC1HWC0", "ori_shape": (2,1,4,4),"ori_format": "NDC1HWC0"},
                    {"shape": (2,1,1,1,1,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (1,),"ori_format": "NDHWC"},
                    {"shape": (2,1,1,1,16,16), "dtype": "float16", "format": "NDC1HWC0", "ori_shape": (2,1,4,4),"ori_format": "NDC1HWC0"}],
         "case_name": "ascend_dequant_6",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case7 = {"params": [{"shape": (1,4,4,16,16), "dtype": "int32", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"},
                    {"shape": (2,1,1,1,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,1,1,1,16),"ori_format": "NC1HWC0"},
                    {"shape": (1,4,4,16,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (1,4,4,16,16),"ori_format": "NC1HWC0"}],
         "case_name": "ascend_dequant_7",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}

def test_matmul_dequant_compute(test_arg):
    x = tvm.placeholder((4, 4, 16, 16), name="matmul_input", attrs={'format': "FRACTAL_NZ", "ori_shape": (64, 64)}, dtype="int32")
    x_shape = (4, 4, 16, 16)
    deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 64, 1, 1)}, dtype="uint64")
    _matmul_compute(x, x_shape, deq_scale, False, False, (64, 64), 0, True)

def test_matmul_dequant_compute_1(test_arg):
    x = tvm.placeholder((1, 4, 1, 16), name="matmul_input", attrs={'format': "NC1HWC0", "ori_shape": (1, 64)}, dtype="int32")
    x_shape = (1, 4, 1, 16)
    deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 64, 1, 1)}, dtype="uint64")
    _matmul_compute(x, x_shape, deq_scale, False, False, (1, 64), 1, True)

def test_dequant_compute_1(test_arg):
    x = tvm.placeholder((1, 4, 1, 16), name="input", attrs={'format': "NDC1HWC0", "ori_shape": (1, 4, 1, 16)}, dtype="int32")
    deq_scale = tvm.placeholder((1, 1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NDC1HWC0", "ori_shape": (1, 1, 4, 1, 1, 16)}, dtype="float16")
    deq_scale_v200 = tvm.placeholder((1, 1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NDC1HWC0", "ori_shape": (1, 1, 4, 1, 1, 16)}, dtype="uint64")
    ascend_dequant_compute(x, deq_scale, None)
    ascend_dequant_compute(x, deq_scale, None, sqrt_mode=True, relu_flag=True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    ascend_dequant_compute(x, deq_scale_v200, None)
    ascend_dequant_compute(x, deq_scale_v200, None, sqrt_mode=True, relu_flag=True)
    cce_conf.cce_conf.te_set_version("Ascend310")


def test_dequant_compute_2(test_arg):
    x = tvm.placeholder((1, 4, 1, 16), name="input", attrs={'format': "NDC1HWC0", "ori_shape": (1, 1, 1, 64)}, dtype="int32")
    deq_scale = tvm.placeholder((1, 1, 1, 1, 1, 16), name="deq_tensor", attrs={'format': "NDC1HWC0", "ori_shape": (1, 1, 1, 1)}, dtype="float16")
    deq_scale_v200 = tvm.placeholder((1, 1, 1, 1, 1, 16), name="deq_tensor", attrs={'format': "NDC1HWC0", "ori_shape": (1, 1, 1, 1)}, dtype="uint64")
    ascend_dequant_compute(x, deq_scale, None)
    ascend_dequant_compute(x, deq_scale, None, sqrt_mode=True, relu_flag=True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    ascend_dequant_compute(x, deq_scale_v200, None)
    ascend_dequant_compute(x, deq_scale_v200, None, sqrt_mode=True, relu_flag=True)
    cce_conf.cce_conf.te_set_version("Ascend310")

def test_dequant_compute_3(test_arg):
    x = tvm.placeholder((1, 4, 1, 16), name="input", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 64)}, dtype="int32")
    deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 64)}, dtype="float16")
    deq_scale_v200 = tvm.placeholder((1, 4, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 64)}, dtype="uint64")
    ascend_dequant_compute(x, deq_scale, None)
    ascend_dequant_compute(x, deq_scale, None, sqrt_mode=True, relu_flag=True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    ascend_dequant_compute(x, deq_scale_v200, None)
    ascend_dequant_compute(x, deq_scale_v200, None, sqrt_mode=True, relu_flag=True)
    cce_conf.cce_conf.te_set_version("Ascend310")

def test_dequant_compute_4(test_arg):
    x = tvm.placeholder((1, 4, 1, 16), name="input", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 64)}, dtype="int32")
    deq_scale = tvm.placeholder((1, 1, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 1)}, dtype="float16")
    deq_scale_v200 = tvm.placeholder((1, 1, 1, 1, 16), name="deq_tensor", attrs={'format': "NC1HWC0", "ori_shape": (1, 1, 1, 1)}, dtype="uint64")
    ascend_dequant_compute(x, deq_scale, None)
    ascend_dequant_compute(x, deq_scale, None, sqrt_mode=True, relu_flag=True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    ascend_dequant_compute(x, deq_scale_v200, None)
    ascend_dequant_compute(x, deq_scale_v200, None, sqrt_mode=True, relu_flag=True)
    cce_conf.cce_conf.te_set_version("Ascend310")

def test_conv2d_rmpad(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from te.platform.fusion_manager import fusion_manager
    from tbe.dsl import auto_schedule
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute
    from impl.relu6 import relu6_compute
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.conv2d_data_rm import conv2d_data_rm_compute
    from tbe.dsl.static_schedule.conv_schedule import AutoScheduleOp

    cce_conf.te_set_version('Ascend310')
    shape_in = (16, 1024, 7, 7)
    shape_w = (1024, 1024, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 31) // 32
    Ci0 = 32

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    with tvm.target.cce():
        fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="int8", attrs={'ori_format': 'NCHW'})

        filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int8",
                                   attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
        bias_tensor = None
        vdeq = tvm.placeholder(shape_scale, name='vreq_reg', dtype="float16",
                               attrs={'ori_shape': [Co1*Co0]})

        conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})
        dequant = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=False)
        relu = relu6_compute(dequant, None)
        out = ascend_quant_compute(relu, None, scale=1, offset=0, sqrt_mode=False)
        out = conv2d_data_rm_compute(out, res_tensor=None)
        tensor_list = [fm, filter_w, vdeq, out]
        sch = auto_schedule(out)

    config = {
        "print_ir": False,
        "need_build": True,
        "name": "conv2d",
        "tensor_list": tensor_list}
    te.lang.cce.cce_build_code(sch, config)
    print("adding Conv2D rmpad ut testcases")

def test_conv2d_vector(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute

    cce_conf.te_set_version('Ascend310')
    shape_in = (16, 1024, 7, 7)
    shape_w = (1024, 1024, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 31) // 32
    Ci0 = 32

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="int8", attrs={'ori_format': 'NCHW',
                                                                         "conv_shape": (16, 1024, 16, 16),
                                                                         "true_conv_shape": shape_in,
                                                                         "invalid_data_rm_flag": 1,
                                                                         "remove_padded_column_in_next_op": 1,
                                                                         "bias_flag":0
                                                                         })

    filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int8",
                               attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
    bias_tensor = None
    vdeq = tvm.placeholder(shape_scale, name='vreq_reg', dtype="float16",
                           attrs={'ori_shape': [Co1*Co0]})

    vdeq_v200 = tvm.placeholder(shape_scale, name="deq_tensor",
                                     attrs={'ori_shape': [Co1*Co0]}, dtype="uint64")

    conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})

    res = _vector_dequant_v100(conv_res, shape_in, (16, 1024, 16, 16), vdeq, True, True, True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    _vector_dequant_v200(conv_res, shape_in, (16, 1024, 16, 16), vdeq_v200, True, True)
    get_conv_flag(conv_res)
    is_conv3d_fuse(conv_res)
    is_support_a100()
    is_support_a100(True)
    get_scale_indices(vdeq, True, Co0, Co1)
    get_depthwise_conv2d_tensor_info(res)
    util.Constant.func_map.get("vdeq_cast")
    util.Constant.func_map.get("deq_cast")
    is_lhisi_version()
    is_support_v200()
    cce_conf.cce_conf.te_set_version("Ascend310")

def test_conv2d_scalar(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute

    cce_conf.te_set_version('Ascend310')
    shape_in = (16, 8, 7, 7)
    shape_w = (8, 8, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 31) // 32
    Ci0 = 32

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="int8", attrs={'ori_format': 'NCHW',
                                                                         "conv_shape": (16, 8, 16, 16),
                                                                         "true_conv_shape": shape_in,
                                                                         "invalid_data_rm_flag": 1,
                                                                         "remove_padded_column_in_next_op": 1
                                                                         })

    filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int8",
                               attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
    bias_tensor = None
    vdeq = tvm.placeholder(shape_scale, name='vreq_reg', dtype="float16",
                           attrs={'ori_shape': [Co1]})

    vdeq_v200 = tvm.placeholder(shape_scale, name="deq_tensor",
                                     attrs={'ori_shape': [Co1]}, dtype="uint64")

    conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})
    _scalar_dequant_v100(conv_res, shape_in, (16, 8, 16, 16), vdeq, True, True, True)

    cce_conf.cce_conf.te_set_version("Ascend310P3")
    _scalar_dequant_v200(conv_res, shape_in, (16, 8, 16, 16), vdeq_v200, True)
    cce_conf.cce_conf.te_set_version("Ascend310")

def test_conv2d_int4(test_arg):
    import sys
    import te.lang.cce
    from te import tvm
    from tbe.common import utils
    from te import platform as cce_conf
    from te import platform as cce
    from impl.conv2d import conv2d_compute

    cce_conf.te_set_version('Ascend310')
    shape_in = (16, 8, 7, 7)
    shape_w = (8, 8, 1, 1)
    pads = (0, 0, 0, 0)
    strides = (1, 1)

    Ni, Ci, Hi, Wi = shape_in
    Co, _, Hk, Wk = shape_w

    Ci1 = (Ci + 63) // 64
    Ci0 = 64

    Co1 = (Co + 15) // 16
    Co0 = 16

    shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
    shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

    shape_scale = (1, Co1, 1, 1, 16)

    dilations = [1, 1, 1, 1]
    strides = [1, 1, strides[0], strides[1]]

    fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype="int4", attrs={'ori_format': 'NCHW',
                                                                         "conv_shape": (16, 8, 16, 16),
                                                                         "true_conv_shape": shape_in,
                                                                         "invalid_data_rm_flag": 1,
                                                                         "remove_padded_column_in_next_op": 1
                                                                         })

    filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype="int4",
                               attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
    bias_tensor = None
    vdeq_v200 = tvm.placeholder(shape_scale, name="deq_tensor",
                                     attrs={'ori_shape': [Co1]}, dtype="uint64")
    conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, None, strides, pads, dilations, offset_x=0, options={"invalid_data_rm": True})
    cce_conf.cce_conf.te_set_version("Ascend310P3")
    _scalar_dequant_v200(conv_res, shape_in, (16, 8, 16, 16), vdeq_v200, True)
    cce_conf.cce_conf.te_set_version("Ascend310")

ut_case.add_case(["Ascend310", "Ascend910"], case1)
ut_case.add_case(["Ascend310", "Ascend910"], case2)
ut_case.add_case(["Ascend310", "Ascend910"], case3)
ut_case.add_case(["Ascend310", "Ascend910"], case4)
ut_case.add_case(["Ascend310", "Ascend910"], case5)
ut_case.add_case(["Ascend310", "Ascend910"], case6)
ut_case.add_case(["Ascend310", "Ascend910"], case7)
ut_case.add_cust_test_func(test_func=test_matmul_dequant_compute)
ut_case.add_cust_test_func(test_func=test_matmul_dequant_compute_1)
ut_case.add_cust_test_func(test_func=test_dequant_compute_1)
ut_case.add_cust_test_func(test_func=test_dequant_compute_2)
ut_case.add_cust_test_func(test_func=test_dequant_compute_3)
ut_case.add_cust_test_func(test_func=test_dequant_compute_4)
ut_case.add_cust_test_func(test_func=test_conv2d_rmpad)
ut_case.add_cust_test_func(test_func=test_conv2d_vector)
ut_case.add_cust_test_func(test_func=test_conv2d_scalar)
ut_case.add_cust_test_func(test_func=test_conv2d_int4)


if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)


