#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3D
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3D", "impl.conv3d", "conv3d")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    fmap={'ori_shape': (1, 8, 60, 88, 32), 'shape': (1, 8, 60, 88, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    weight={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    bias=None, offset_w=None,
    output={'ori_shape': (1, 4, 30, 44, 64), 'shape': (1, 4, 30, 44, 64),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC", offset_x=0):
    return [fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, offset_x]


def test_op_check_supported(test_arg):
    from impl.conv3d import check_supported
    fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    (fmap, weight, bias, offset_w, output, strides,
            pads, dilations, groups, data_format, _) = _run_api_end_with_d(fmap = fmap)
    check_supported(fmap, weight, bias, offset_w, output, strides, pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

# test_conv3dbp_succ_d
case1 = _run_api_end_with_d()

# test_conv3dbp_stride_one
fmap = {'ori_shape': (1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'},
weight = {'ori_shape': (64, 32, 2, 2, 2), 'shape': (64, 32, 2, 2, 2),
          'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'},
output = {'ori_shape': (1, 7, 59, 87, 64), 'shape': (1, 7, 59, 87, 64),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
strides = (1, 1, 1, 1, 1)
case2 = _run_api_end_with_d(output=output, strides=strides)

# test_bias_length_fail
bias = {'ori_shape': (64, 64), 'shape': (64, 64),
        'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
case3 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (2, 32, 15, 4098, 18), 'shape': (2, 32, 15, 4098, 18),
        'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
case4 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_output
output = {'dtype': 'float32'}
case5 = _run_api_end_with_d(output=output)

# test_conv3d_invalid_dilations
dilations = (1, 2, 1, 1, 1)
case6 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case7 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_pad_length
pads = (0, -1, -1, -1, 0)
case8 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32), 'shape': (2, 2, 2, 32),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case9 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight_D
weight = {'ori_shape': (2, 2, 354, 32, 64), 'shape': (2, 2, 354, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case10 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_big_fmap
fmap = {'ori_shape': (200, 3000, 4000, 4000, 3000),
      'shape': (200, 3000, 4000, 4000, 3000),
      'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case11 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_bias_dtype
bias = {'ori_shape': (1,), "dtype": "float32"}
case12 = _run_api_end_with_d(bias=bias)

# test_conv3d_invalid_pads
pads = (1, 1, 1, 1, 3, 1)
case13 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 0, 1, 1)
case14 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_fmap_format
fmap = {'ori_shape': (1, 32, 8, 60, 88), 'shape': (1, 32, 8, 60, 88),
      'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
case15 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
        'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float16'}
case16 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_stride_shape
strides = (1, 0, 1, 1, 1)
case17 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 0, 1, 1)
case18 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_stride_shape
strides = (1, 1, 1, 0, 1)
case19 = _run_api_end_with_d(strides=strides)

# test_conv3d_invalid_weight
weight = {'ori_shape': (257, 2, 2, 32, 64), 'shape': (257, 2, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case20 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 257, 2, 32, 64), 'shape': (2, 257, 2, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case21 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_dilations
dilations = (1, 0, 1, 1, 1)
case22 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_pad_length
pads = (256, 256, 256, 256, 256, 256)
case23 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_fmap_shape
fmap = {'ori_shape': (1, 8, 60, 4098, 32), 'shape': (1, 8, 60, 4098, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case24 = _run_api_end_with_d(fmap=fmap)

fmap = {'ori_shape': (1, 8, 4098, 88, 32), 'shape': (1, 8, 4098, 88, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case25 = _run_api_end_with_d(fmap=fmap)

# test_conv3d_invalid_weight
weight = {'ori_shape': (2, 2, 257, 32, 64), 'shape': (2, 2, 257, 32, 64),
        'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case26 = _run_api_end_with_d(weight=weight)

# test_conv3d_invalid_pad_length
pads = (3, 3, 256, 256, 256, 256)
case27 = _run_api_end_with_d(pads=pads)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case1, "success", "case1", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case2, "success", "case2", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case3, RuntimeError, "case3", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case4, RuntimeError, "case4", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case5, RuntimeError, "case5", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case6, RuntimeError, "case6", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case7, RuntimeError, "case7", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case8, RuntimeError, "case8", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case9, RuntimeError, "case9", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case10, RuntimeError, "case10", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case11, RuntimeError, "case11", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case12, RuntimeError, "case12", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case13, RuntimeError, "case13", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case14, RuntimeError, "case14", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case15, RuntimeError, "case14", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case16, RuntimeError, "case14", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case17, RuntimeError, "dynamic_case17", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case18, RuntimeError, "dynamic_case18", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case19, RuntimeError, "dynamic_case19", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case20, RuntimeError, "dynamic_case20", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case21, RuntimeError, "dynamic_case21", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case22, RuntimeError, "dynamic_case22", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case23, RuntimeError, "dynamic_case23", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case24, RuntimeError, "dynamic_case24", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case25, RuntimeError, "dynamic_case25", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case26, RuntimeError, "dynamic_case26", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case27, RuntimeError, "dynamic_case27", True))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
