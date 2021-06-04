#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropFilterD", "impl.conv3d_backprop_filter_d",
               "conv3d_backprop_filter_d")


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api_end_with_d(
    x_dict={'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
            'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'},
    out_backprop={'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                  'ori_format': 'NDHWC', 'format': 'NDHWC',
                  'dtype': 'float16'},
    y_input={'ori_shape': (2, 2, 2, 32, 64), 'shape': (2, 2, 2, 32, 64),
             'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'},
    filter_size=(2, 2, 2, 32, 64),
    strides=(1, 2, 2, 2, 1),
    pads=(0, 0, 0, 0, 0, 0),
    dilations=(1, 1, 1, 1, 1),
    groups=1,
    data_format="NDHWC"):
    return [x_dict, out_backprop, y_input, filter_size, strides,
            pads, dilations, groups, data_format]


def test_op_check_supported(test_arg):
    from impl.conv3d_backprop_filter_d import check_supported
    out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
    (x_dict, out_backprop, y_input, filter_size, strides, pads, dilations,
        groups, data_format) = _run_api_end_with_d(out_backprop=out_backprop)
    check_supported(x_dict, out_backprop, y_input, filter_size, strides, pads, dilations, groups, data_format)


ut_case.add_cust_test_func(test_func=test_op_check_supported)

def _test_op_get_op_support_info(test_arg):
    from impl.conv3d_backprop_filter_d import get_op_support_info

    [x_dict, out_backprop, y_input, filter_size, strides,
     pads, dilations, groups, data_format] = _run_api_end_with_d()

    get_op_support_info(
        x_dict, out_backprop, y_input, filter_size, strides,
        pads, dilations, groups, data_format)

ut_case.add_cust_test_func(test_func=_test_op_get_op_support_info)

# Define Cases instances
# test_conv3dbp_filter_succ
case1 = _run_api_end_with_d()

# test_conv3dbp_filter_stride_one
strides = (1, 1, 1)
out_backprop = {'ori_shape': (1, 15, 119, 175, 64),
                'shape': (1, 15, 119, 175, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case2 = _run_api_end_with_d(out_backprop=out_backprop, strides=strides)

# test_conv3dbp_filter_x_dict_failed
out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case3 = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_invalid_pads
pads = [0, 0, 0, 0]
case4 = _run_api_end_with_d(pads=pads)

# test_conv3dbp_filter_invalid_dilations
dilations = [1, 0, 1, 0]
case5 = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_filter_invalid_shape
y_input = {'ori_shape': (2, 2, 16, 64), 'shape': (2, 2, 16, 64),
           'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float16'}
case6 = _run_api_end_with_d(y_input=y_input)

x_dict = {'ori_shape': (16, 120, 176, 32), 'shape': (16, 120, 176, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case7 = _run_api_end_with_d(x_dict=x_dict)

# test_conv3dp_filter_invalid_input_type
x_dict = {'ori_shape': (1, 16, 120, 176, 32), 'shape': (1, 16, 120, 176, 32),
          'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float32'}
case8 = _run_api_end_with_d(x_dict=x_dict)

out_backprop = {'ori_shape': (1, 8, 60, 88, 64), 'shape': (1, 8, 60, 88, 64),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float32'}
case9 = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_invalid_stride_range_pad_SAME
padding = 'NOTSAME'
case10 = _run_api_end_with_d(pads=padding)

# test_conv3dbp_filter_invalid_shape_length
out_backprop = {'ori_shape': (1, 8, 60, 88), 'shape': (1, 8, 60, 88),
                'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case11 = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_invalid_stides_length
strides = (1, 1, 1, 1)
case12 = _run_api_end_with_d(strides=strides)

# test_conv3dbp_filter_invalid_filter_size_length
filter_size = (2, 2, 2, 64)
case13 = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_invalid_pads_length
pads = {"2": 2}
case14 = _run_api_end_with_d(pads=pads)

# test_conv3dbp_filter_invalid_dilations_value
dilations = [2, 1, 1, 1, 2]
case15 = _run_api_end_with_d(dilations=dilations)

# test_conv3dbp_filter_x_format_failed
out_backprop = {'ori_shape': (1, 5, 19, 75, 64), 'shape': (1, 5, 19, 75, 64),
                'ori_format': 'NDCHW', 'format': 'NDCHW', 'dtype': 'float32'}
case16 = _run_api_end_with_d(out_backprop=out_backprop)

# test_conv3dbp_filter_wrong_filter_batch
filter_size = (2, 2, 2, 32, 32)
case17 = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_wrong_filter_channel
filter_size = (2, 2, 2, 22, 64)
case18 = _run_api_end_with_d(filter_size=filter_size)

# test_conv3dbp_filter_wrong_fmap_batch
x_dict = {'ori_shape': (2, 16, 120, 176, 32), 'shape': (2, 16, 120, 176, 32),
        'ori_format': 'NDHWC', 'format': 'NDHWC', 'dtype': 'float16'}
case19 = _run_api_end_with_d(x_dict=x_dict)


out_backprop={'ori_shape': (1, 9, 61, 89, 64), 'shape': (1, 9, 61, 89, 64),
              'ori_format': 'NDHWC', 'format': 'NDHWC',
              'dtype': 'float16'}
pads=(1, 1, 1, 1, 1, 1)
case20 = _run_api_end_with_d(out_backprop=out_backprop, pads=pads)

# test_conv3dbp_filter Not flag all one
x_dict={'ori_shape': (1, 32, 240, 352, 16), 'shape': (1, 32, 240, 352, 16),
              'ori_format': 'NDHWC', 'format': 'NDHWC',
              'dtype': 'float16'}
out_backprop={'ori_shape': (1, 32, 240, 352, 16), 'shape': (1, 32, 240, 352, 16),
            'ori_format': 'NDHWC', 'format': 'NDHWC',
            'dtype': 'float16'}
y_input = {'ori_shape': (3, 3, 3, 16, 16), 'shape': (3, 3, 3, 16, 16),
            'ori_format': 'DHWCN', 'format': 'DHWCN',
            'dtype': 'float32'}
filter_size = (3, 3, 3, 16, 16)
pads=(1, 1, 1, 1, 1, 1)
strides=(1,1,1,1,1)
case21 = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                             y_input=y_input, filter_size=filter_size, pads=pads, strides=strides)

x_dict={'ori_shape': (4,5,49,14,964), 'shape': (4,5,49,14,964),
              'ori_format': 'NDHWC', 'format': 'NDHWC',
              'dtype': 'float16'}
out_backprop={'ori_shape': (4,2,9,2,185), 'shape': (4,2,9,2,185),
            'ori_format': 'NDHWC', 'format': 'NDHWC',
            'dtype': 'float16'}
y_input = {'ori_shape': (2, 2, 6, 964, 185), 'shape': (2, 2, 6, 964, 185),
            'ori_format': 'DHWCN', 'format': 'DHWCN',
            'dtype': 'float32'}
filter_size = (2, 2, 6, 964, 185)
pads=(0, 0, 8, 9, 5, 5)
strides=(1,1,17,3,1)
case22 = _run_api_end_with_d(x_dict=x_dict, out_backprop=out_backprop,
                             y_input=y_input, filter_size=filter_size, pads=pads, strides=strides)
# Add test Cases
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

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case19, RuntimeError, "case19", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case20, "success", "case20", True))

ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case21, "success", "case20", True))
ut_case.add_case(["Ascend910", "Ascend310"],
                 _gen_data_case(case22, "success", "case20", True))
if __name__ == '__main__':
    ut_case.run()
    exit(0)
