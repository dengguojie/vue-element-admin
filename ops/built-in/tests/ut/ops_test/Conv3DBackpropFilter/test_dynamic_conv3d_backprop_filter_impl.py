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

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
