#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropInput", "impl.dynamic.conv3d_backprop_input", "conv3d_backprop_input")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api(
        input_size={'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'},
        filter={'ori_shape': (1, 1, 1, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'},
        out_backprop={'ori_shape': (1, 8, -1, -1, 64), 'ori_format': 'NDHWC', 'dtype': 'float16',
                      'range': ((1, 1), (8, 8), (4, 4), (1, 56), (1, 75), (16, 16))},
        y={'ori_shape': (1, 8, -1, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
           'range': ((1, 1), (8, 8), (16, 16), (1, 56), (1, 75), (16, 16))},
        strides=(1, 1, 1, 1, 1),
        pads=[0, 0, 0, 0, 0, 0],
        dilations=(1, 1, 1, 1, 1),
        groups=1,
        data_format="NDHWC"):
    return [input_size, filter, out_backprop, y, strides, pads, dilations, groups, data_format]


def test_conv3d_backprop_input_fuzz_build_generalization(test_arg):
    from impl.dynamic.conv3d_backprop_input import conv3d_backprop_input_generalization
    input_list = [
        {
            'shape': (5,),
            'ori_shape': (5,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, {
            'ori_shape': (1, 1, 1, 256, 64),
            'ori_format': 'DHWCN',
            'format': 'FRACTAL_Z_3D',
            'dtype': 'float16'
        }, {
            'shape': (1, 8, 4, 56, 56, 16),
            'ori_shape': (1, 8, 56, 56, 64),
            'ori_format': 'NDHWC',
            'format': 'NDC1HWC0',
            'dtype': 'float16',
            'range': [(1, 1), (8, 15), (4, 4), (32, 63), (32, 63), (16, 16)]
        }, {
            'shape': (1, 8, 16, 56, 56, 16),
            'ori_shape': (1, 8, 56, 56, 256),
            'ori_format': 'NDHWC',
            'format': 'NDC1HWC0',
            'dtype': 'float16'
        }, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 'conv3d_backprop_input_generalization']
    conv3d_backprop_input_generalization(*input_list)


def test_get_pad_mode(test_args):
    from impl.dynamic.conv3d_backprop_input import _get_pad_mode
    _get_pad_mode([-1,-1,-1,-1,-1,-1])
    _get_pad_mode("SAME")

ut_case.add_cust_test_func(test_func=test_get_pad_mode)

# test_conv3dbp_succ_dynamic
case1 = _run_api()

# test_conv3dbp_succ_dynamic_same_pad_depthwise
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (3, 6, 17, 1, 16), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 16, -1, -1, 9), 'ori_format': 'NCDHW', 'dtype': 'float16',
                'range': ((1, 3), (12, 12), (2, 2), (5, 40), (9, 9), (16, 16))}
y = {'ori_shape': (-1, 8, -1, -1, 26), 'ori_format': 'NCDHW', 'dtype': 'float16',
     'range': ((1, 3), (24, 24), (1, 1), (16, 68), (26, 26), (16, 16))}
strides = (1, 1, 2, 3, 3)
pads = [-1, -1, -1, -1, -1, -1]
groups = 8
data_format="NCDHW"

case2 = _run_api(input_size, filter, out_backprop, y, strides, pads, groups=groups, data_format=data_format)

# test_conv3dbp_exception_error_pad
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (6, 3, 10, 6, 72), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 85, 25, 5, 72), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, 2), (85, 85), (5, 5), (25, 25), (5, 5), (16, 16))}
y = {'ori_shape': (-1, 423, 73, 16, 6), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 2), (73, 73), (1, 1), (73, 73), (16, 16), (16, 16))}
strides = (1, 5, 3, 5, 1)
pads = [2, 10, 1, 1, 4, 5]
case3 = _run_api(input_size, filter, out_backprop, y, strides, pads)

# test_conv3dbp_exception_error_dilation
dilations = [1, 2, 1, 1, 1]
case4 = _run_api(dilations=dilations)

# test_conv3dbp_exception_-2
input_size = {'ori_shape': (-2,), 'ori_format': 'ND', 'dtype': 'int32'}
case5 = _run_api(input_size=input_size)

# test filter worng format
filter = {'ori_shape': (6, 3, 10, 6, 72), 'ori_format': 'ND', 'dtype': 'float16'}
case6 = _run_api(filter=filter)

# test NCDHW filter
filter = {'ori_shape': (6, 3, 10, 6, 72), 'ori_format': 'NCDHW', 'dtype': 'float16'}
case7 = _run_api(filter=filter)

# test NDHWC filter
filter = {'ori_shape': (6, 3, 10, 6, 72), 'ori_format': 'NDHWC', 'dtype': 'float16'}
case8 = _run_api(filter=filter)

# test out_backprop worng format
out_backprop = {'ori_shape': (-1, 85, 25, 5, 72), 'ori_format': 'ND', 'dtype': 'float16'}
case9 = _run_api(out_backprop=out_backprop)

# test -1 in range C
y = {'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 1), (8, 8), (56, 56), (1, 75), (-1, 16))}
case10 = _run_api(y=y)

# test -1 in range
y = {'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'ND', 'dtype': 'float16',
     'range': ((1, 1), (8, 8), (56, 56), (-1, 75), (16, 16))}
case11 = _run_api(y=y)

# test None in range
y = {'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 1), (8, 8), (56, 56), (75, None), (16, 16))}
case12 = _run_api(y=y)

# test range exceed 4096
y = {'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 1), (8, 8), (56, 56), (75, 5000), (16, 16))}
case13 = _run_api(y=y)

# test range low > high
y = {'ori_shape': (1, 8, 56, -1, 256), 'ori_format': 'NDHWC', 'dtype': 'float16',
     'range': ((1, 1), (8, 8), (56, 56), (75, 2), (16, 16))}
case14 = _run_api(y=y)

# test filter_d_dilation > fmap_d_padding
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (20, 5, 5, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 8, 8, 8, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, 1), (8, 8), (1, 1), (8, 8), (8, 8), (16, 16))}
y = {'ori_shape': (-1, 16, 16, 16, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
    'range': ((1, 1), (16, 16), (1, 1), (16, 16), (16, 16), (16, 16))}
strides = (1, 2, 2, 2, 1)
pads = [1, 2, 1, 2, 1, 2]
case15 = _run_api(out_backprop=out_backprop, y=y, strides=strides,
                  input_size=input_size, filter=filter, pads=pads)

# test filter_h_dilation > fmap_h_padding
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (5, 20, 5, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 8, 8, 8, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, 1), (8, 8), (1, 1), (8, 8), (8, 8), (16, 16))}
y = {'ori_shape': (-1, 16, 16, 16, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
    'range': ((1, 1), (16, 16), (1, 1), (16, 16), (16, 16), (16, 16))}
strides = (1, 2, 2, 2, 1)
pads = [1, 2, 1, 2, 1, 2]
case16 = _run_api(out_backprop=out_backprop, y=y, strides=strides,
                  input_size=input_size, filter=filter, pads=pads)

# test filter_w_dilation > fmap_w_padding
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (5, 5, 50, 256, 64), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-1, 8, 8, 8, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, 1), (8, 8), (1, 1), (8, 8), (8, 8), (16, 16))}
y = {'ori_shape': (-1, 16, 16, 16, 16), 'ori_format': 'NDHWC', 'dtype': 'float16',
    'range': ((1, 1), (16, 16), (1, 1), (16, 16), (16, 16), (16, 16))}
strides = (1, 2, 2, 2, 1)
pads = [1, 2, 1, 2, 1, 2]
case17 = _run_api(out_backprop=out_backprop, y=y, strides=strides,
                  input_size=input_size, filter=filter, pads=pads)

# test dedy_shape = [-2]
input_size = {'ori_shape': (5,), 'ori_format': 'ND', 'dtype': 'int32'}
filter = {'ori_shape': (3, 3, 3, 128, 256), 'ori_format': 'DHWCN', 'dtype': 'float16'}
out_backprop = {'ori_shape': (-2,), 'ori_format': 'NDHWC', 'dtype': 'float16',
                'range': ((1, None), (1, None), (16, 16), (1, None), (1, None), (16, 16))}
y = {'ori_shape': (-1, -1, -1, -1, 128), 'ori_format': 'NDHWC', 'dtype': 'float16',
    'range': ((1, None), (1, None), (8, 8), (1, None), (1, None), (16, 16))}
strides = (1, 2, 2, 2, 1)
pads = [1, 2, 1, 2, 1, 2]
case18 = _run_api(out_backprop=out_backprop, y=y, strides=strides,
                  input_size=input_size, filter=filter, pads=pads)

def test_conv3d_bp_input_fuzz_build_tilingcase(test_arg):
    import json
    from impl.dynamic.conv3d_backprop_input import conv3d_backprop_input
    from tbe.common.context import get_context
    from tbe.common.context import op_context
    with op_context.OpContext("dynamic"):
        get_context().set_build_type("fuzzily_build")
        get_context().add_addition("max_kernel_id", -1)
        missing_info = [{
                            "inputs": [{
                                "index": 2,
                                "tensor": [{
                                    "range": [
                                        [2, 4],
                                        [8, 16],
                                        [8, 16],
                                        [8, 16],
                                        [16, 16]
                                    ],
                                    "shape": [-1, -1, -1, -1, 16]
                                }]
                            }]
                        }]
        get_context().add_addition("missing_support_info", json.dumps(missing_info))
        input_list = [
            {
            'shape': (5,),
            'ori_shape': (5,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
            }, {
                'shape': (8, 1, 16, 16),
                'ori_shape': (2, 2, 2, 16, 16),
                'ori_format': 'DHWCN',
                'format': 'FRACTAL_Z_3D',
                'dtype': 'float16'
            }, {
                'shape': (2, 8, 1, 8, 8, 16),
                'ori_shape': (-1, -1, -1, -1, 16),
                'ori_format': 'NDHWC',
                'format': 'NDC1HWC0',
                'dtype': 'float16',
                'range': [(2, 3), (8, 15), (1, 1), (8, 15), (8, 15), (16, 16)]
            }, {
                'shape': (2, 9, 1, 9, 9, 16),
                'ori_shape': (-1, -1, -1, -1, 16),
                'ori_format': 'NDHWC',
                'format': 'NDC1HWC0',
                'dtype': 'float16',
                'range': [(2, 3), (8, 15), (1, 1), (8, 15), (8, 15), (16, 16)]
            }, (1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1), 1, 'NDHWC', 'conv3d_dx_generalization']
        conv3d_backprop_input(*input_list)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case1", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case2_same_pad_depthwise", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, RuntimeError, "dynamic_case3_error_padding_d", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, RuntimeError, "dynamic_case4_error_dilation", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, RuntimeError, "dynamic_case5_error_minus_2", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case6, RuntimeError, "dynamic_case6", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case7, RuntimeError, "dynamic_case7", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case8, RuntimeError, "dynamic_case8", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case9, RuntimeError, "dynamic_case9", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case10, RuntimeError, "dynamic_case10", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case11, RuntimeError, "dynamic_case11", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case12, RuntimeError, "dynamic_case12", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case13, RuntimeError, "dynamic_case13", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case14, RuntimeError, "dynamic_case14", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case15, RuntimeError, "dynamic_case15", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case16, RuntimeError, "dynamic_case16", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case17, RuntimeError, "dynamic_case17", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case18, RuntimeError, "dynamic_case18", True))

# test_conv3d_backprop_input_fuzz_build_generalization
print("adding conv3d test_conv3d_backprop_input_fuzz_build_generalization testcase")
ut_case.add_cust_test_func(test_func=test_conv3d_backprop_input_fuzz_build_generalization)

print("add test_conv3d_bp_input_fuzz_build_tilingcase")
ut_case.add_cust_test_func(test_func=test_conv3d_bp_input_fuzz_build_tilingcase)

if __name__ == '__main__':
    ut_case.run()
