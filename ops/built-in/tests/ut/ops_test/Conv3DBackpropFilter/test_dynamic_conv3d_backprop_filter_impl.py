#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("Conv3DBackpropFilter", "impl.dynamic.conv3d_backprop_filter", "conv3d_backprop_filter")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


def _run_api(x_shape, x_format, x_dtype, x_range,
             filter_shape, filter_format, filter_dtype, filter_range,
             out_shape, out_format, out_dtype, out_range,
             strides=(1, 1, 1, 1, 1),
             pads=[0, 0, 0, 0, 0, 0],
             dilations=(1, 1, 1, 1, 1),
             groups=1,
             data_format="NDHWC"):
    x = {'ori_shape': x_shape, 'shape': x_shape,
         'ori_format': x_format, 'format': x_format, 'dtype': x_dtype, 
         "range": x_range}
    filter = {'ori_shape': filter_shape, 'shape': filter_shape,
              'ori_format': filter_format, 'format': filter_format, 'dtype': filter_dtype, 
              "range": filter_range}
    filter_size = filter
    out_backprop = {'ori_shape': out_shape, 'shape': out_shape,
                    'ori_format': out_format, 'format': out_format, 'dtype': out_dtype, 
                    "range": out_range}

    return [x, filter_size, out_backprop, filter, strides, pads, dilations, groups, data_format]

def _run_api_v2(
    x=None, filter=None, out_backprop=None, strides=(1, 1, 1, 1, 1),
    pads=[0, 0, 0, 0, 0, 0], dilations=(1, 1, 1, 1, 1), groups=1, data_format="NDHWC"):
    if x is None:
        x = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, -1, -1, 256),
             'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
             "range": [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)]}
    if filter is None:
        filter = {'ori_shape': (3, 3, 3, 256, 256), 'shape': (3, 3, 3, 256, 256),
                  'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32', 
                  "range": [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)]}
    if out_backprop is None:
        out_backprop = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, -1, -1, 256),
                        'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                        "range": [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)]}
    filter_size = filter
    return [x, filter_size, out_backprop, filter, strides, pads, dilations, groups, data_format]

# test_conv3dbp_succ_dynamic
case1 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)])

case2 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid pads, D/H/W is not dynamic
case3 = _run_api((-1, 5, 126, 126, 128), "NDHWC", "float16",
                 [(1, 50), (5, 5), (126, 126), (126, 126), (128, 128)],
                 (3, 3, 3, 128, 128), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (128, 128), (128, 128)],
                 (-1, 3, 124, 124, 128), "NDHWC", "float16",
                 [(1, 50), (3, 3), (124, 124), (124, 124), (128, 128)],
                 [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1])

# invalid pads, pads has -2
case4 = _run_api((-1, -1, 126, 126, 128), "NDHWC", "float16",
                 [(1, 50), (5, 50), (126, 126), (126, 126), (128, 128)],
                 (3, 3, 3, 128, 128), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (128, 128), (128, 128)],
                 (-1, 3, 124, 124, 128), "NDHWC", "float16",
                 [(1, 50), (3, 3), (124, 124), (124, 124), (128, 128)],
                 [1, 1, 1, 1, 1], [-2, -1, -1, -1, -1, -1])

# invalid data_format, data_format is not same as Fmap's format
case5 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 data_format='NCDHW')

# invalid fmap shape d_dim = -2
case6 = _run_api((1, -2, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid fmap shape h_dim = 0
case7 = _run_api((1, -1, 0, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid fmap shape
case8 = _run_api((1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid dilations
case9 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 dilations=(2, 1, 1, 1, 1))

# invalid groups
case10 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 groups=2)

case11 = _run_api((-2,), "NDHWC", "float16",
                 [(1, None), (1, None), (1, None), (1, None), (1, None)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (-2,), "NDHWC", "float16",
                 [(1, None), (1, None), (1, None), (1, None), (1, None)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 groups=1)

# test pad_d < filter_d constraint
invalid_pads = [4, 4, 0, 0, 0, 0]
case12 = _run_api_v2(pads=invalid_pads)

# test pad_h < filter_h constraint
invalid_pads = [0, 0, 0, 4, 0, 0]
case13 = _run_api_v2(pads=invalid_pads)

# test pad_w < filter_w constraint
invalid_pads = [0, 0, 0, 0, 4, 4]
case14 = _run_api_v2(pads=invalid_pads)

# test convolution result Failed in D direction
x = {'ori_shape': (1, 3, -1, -1, 256), 'shape': (1, 3, -1, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (3, 3), (1, 64), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, 2, -1, -1, 256), 'shape': (1, 2, -1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (2, 2), (1, 152), (10, 10), (256, 256)]}
case15 = _run_api_v2(x=x,out_backprop=out_backprop)

# test convolution result Failed in H direction
x = {'ori_shape': (1, -1, 3, -1, 256), 'shape': (1, -1, 3, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (3, 3), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, -1, 2, -1, 256), 'shape': (1, -1, 2, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (2, 2), (10, 10), (256, 256)]}
case16 = _run_api_v2(x=x,out_backprop=out_backprop)

# test convolution result Failed in W direction
x = {'ori_shape': (1, -1, -1, 4, 256), 'shape': (1, -1, -1, 4, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (1, 64), (4, 4), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 5, 256), 'shape': (1, -1, -1, 5, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (1, 152), (5, 5), (256, 256)]}
case17 = _run_api_v2(x=x,out_backprop=out_backprop)

# test Chip Design demand dedy_w must >=2 when dedy_h != 1
x = {'ori_shape': (1, -1, -1, 3, 256), 'shape': (1, -1, -1, 3, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (1, 64), (3, 3), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 2, 256), 'shape': (1, -1, -1, 2, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (70, 152), (2, 2), (256, 256)]}
case18 = _run_api_v2(x=x,out_backprop=out_backprop)

# test_fail fmap h_lower too small
x = {'ori_shape': (1, -1, 2, -1, 256), 'shape': (1, -1, 2, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (2, 2), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 2, 256), 'shape': (1, -1, -1, 2, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (70, 152), (2, 2), (256, 256)]}
case19 = _run_api_v2(x=x,out_backprop=out_backprop)

# test h and w range correction for Fmap
x = {'ori_shape': (1, -1, 3, -1, 256), 'shape': (1, -1, 3, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (2, 10), (3, 3), (2, 10), (256,256)]}
out_backprop = {'ori_shape': (1, -1, 1, -1, 256), 'shape': (1, -1, 1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (3, 3), (10, 10), (256, 256)]}
case20 = _run_api_v2(x=x,out_backprop=out_backprop)

# NDC1HWC0 and fmap_n != dedy_n
x = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, 16, -1, -1, 16),
     'ori_format': "NDHWC", 'format': "NDC1HWC0", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (16, 16), (1, 64), (20, 20), (16,16)]}
out_backprop = {'ori_shape': (2, -1, -1, -1, 256), 'shape': (2, -1, -1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(2, 2), (26, 38), (1, 152), (10, 10), (256, 256)]}
case21 = _run_api_v2(x=x,out_backprop=out_backprop)
# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_pass_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_SAME_strides_2_2_2_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, RuntimeError, "dynamic_invalid_pads_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, RuntimeError, "dynamic_invalid_pad_2_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, RuntimeError, "dynamic_invalid_format_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case6, RuntimeError, "dynamic_invalid_fmap_d_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case7, RuntimeError, "dynamic_invalid_fmap_h_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case8, RuntimeError, "dynamic_invalid_fmap_dims_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case9, RuntimeError, "dynamic_invalid_dilations_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case10, RuntimeError, "dynamic_invalid_groups_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case11, "success", "dynamic_shape_2_case", True))
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
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case19, RuntimeError, "dynamic_case19", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case20, "success", "dynamic_case20", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case21, RuntimeError, "dynamic_case21", True))
if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
