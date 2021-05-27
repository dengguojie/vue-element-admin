#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInputD
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropInputD", "impl.conv3d_backprop_input_d",
               "conv3d_backprop_input_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    filters={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
             'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'},
    out_backprop={'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                  'ori_format': 'NDHWC', 'format': 'NDHWC',
                  'dtype': 'float16'},
    y_input={'ori_shape': (1, 16, 120, 176, 32),
             'shape': (1, 16, 120, 176, 32),
             'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    input_sizes=(1, 16, 120, 176, 32),
    strides=(1, 2, 2, 2, 1),
    pads=[0, 0, 0, 0, 0, 0],
    dilations=(1, 1, 1, 1, 1),
    groups=1, data_format="NDHWC"):
    return [filters, out_backprop, y_input, input_sizes, strides,
            pads, dilations, groups, data_format]


def test_op_check_supported(test_arg):
    from impl.conv3d_backprop_input_d import check_supported
    input_sizes = (1, 16, 120, 176, 16)
    (filters, out_backprop, y_input, input_sizes, strides,
                    pads, dilations, groups, data_format) = _run_api_end_with_d(input_sizes=input_sizes)
    check_supported(filters, out_backprop, y_input, input_sizes, strides,
                                pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d_backprop_input_d import get_op_support_info

    [filters, out_backprop, y_input, input_sizes, strides,
     pads, dilations, groups, data_format] = _run_api_end_with_d()

    get_op_support_info(
        filters, out_backprop, y_input, input_sizes, strides,
        pads, dilations, groups, data_format)

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)


# test_conv3dbp_succ_d
case1 = _run_api_end_with_d()

# test_conv3dbp_NCDHW
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (1, 64, 8, 60, 88), 'shape': (1, 64, 8, 60, 88),
                'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
y_input = {'ori_shape': (1, 32, 16, 120, 176), 'shape': (1, 32, 16, 120, 176),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
input_sizes = (1, 32, 16, 120, 176)
strides = (1, 1, 2, 2, 2)
case2 = _run_api_end_with_d(
    filters=filters, out_backprop=out_backprop, y_input=y_input,
    input_sizes=input_sizes, strides=strides, data_format='NCDHW')

# test_conv3dbp_invalid_shape
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DHCWN', 'format': 'DHCWN', 'dtype': 'float16'}
case3 = _run_api_end_with_d(filters=filters)

# test_conv3dbp_pad_same_failed
pads = "SAME"
case4 = _run_api_end_with_d(pads=pads)

# test_conv3dbp_outbackprop_wrong_format
out_backprop = {'ori_shape': (8, 60, 88, 64, 1), 'shape': (8, 60, 88, 64, 1),
                'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case5 = _run_api_end_with_d(out_backprop=out_backprop)

# Invalid pads
pads = [0, 0, 0, 0]
case6 = _run_api_end_with_d(pads=pads)

# test_conv3dbp_invalid_dilations
dilations = [1, 0, 1, 0]
case7 = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_invalid_shape
# fmap_channel != filter_channel
input_sizes = (1, 16, 120, 176, 16)
case8 = _run_api_end_with_d(input_sizes=input_sizes)

# dedy_channel != filter_batch
filters = {'ori_shape': (2, 2, 2, 32, 32), 'shape': (2, 2, 2, 32, 32),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case9 = _run_api_end_with_d(filters=filters)

# fmap_batch != dedy_batch
input_sizes = (2, 16, 120, 176, 32)
case10 = _run_api_end_with_d(input_sizes=input_sizes)

# Filter with NDHWC but failed.
filters = {'ori_shape': (64, 2, 2, 2, 32), 'shape': (64, 2, 2, 2, 32),
           'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case11 = _run_api_end_with_d(filters=filters)

# Wrong out_backprop shape
out_backprop = {'ori_shape': (8, 6, 8, 6, 1), 'shape': (8, 6, 8, 6, 1),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case12 = _run_api_end_with_d(out_backprop=out_backprop)

# Wrong dataformat.
data_format = "DHWNC"
case13 = _run_api_end_with_d(data_format=data_format)

# Wrong filter data format
filters = {'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
           'ori_format': 'DNHCW', 'format': 'DNHCW', 'dtype': 'float16'}
case14 = _run_api_end_with_d(filters=filters)

# Wrong y data format
y_input = {'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
           'ori_format': 'DNHCW', 'format': 'DNHCW', 'dtype': 'float16'}
case15 = _run_api_end_with_d(y_input=y_input)

# Wrong dilations
dilations = [2, 1, 1, 1, 2]
case16 = _run_api_end_with_d(dilations=dilations)

# test_conv3d_invalid_pads_dtype
pads = {"2": 2}
case17 = _run_api_end_with_d(pads=pads)

# test_conv3d_invalid_input_sizes
input_sizes = (1, 35, 6, 12, 176)
case18 = _run_api_end_with_d(input_sizes=input_sizes)

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
                 _gen_data_case(case15, RuntimeError, "case15", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case16, RuntimeError, "case16", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case17, RuntimeError, "case17", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case18, RuntimeError, "case18", True))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
