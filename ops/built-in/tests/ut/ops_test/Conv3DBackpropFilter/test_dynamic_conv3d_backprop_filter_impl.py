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
success_case = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)])

success_case2 = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid pads, D/H/W is not dynamic
dynamic_invalid_pads_case = _run_api((-1, 5, 126, 126, 128), "NDHWC", "float16",
                 [(1, 50), (5, 5), (126, 126), (126, 126), (128, 128)],
                 (3, 3, 3, 128, 128), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (128, 128), (128, 128)],
                 (-1, 3, 124, 124, 128), "NDHWC", "float16",
                 [(1, 50), (3, 3), (124, 124), (124, 124), (128, 128)],
                 [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1])

# invalid pads, pads has -2
dynamic_invalid_pad_2_case = _run_api((-1, -1, 126, 126, 128), "NDHWC", "float16",
                 [(1, 50), (5, 50), (126, 126), (126, 126), (128, 128)],
                 (3, 3, 3, 128, 128), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (128, 128), (128, 128)],
                 (-1, 3, 124, 124, 128), "NDHWC", "float16",
                 [(1, 50), (3, 3), (124, 124), (124, 124), (128, 128)],
                 [1, 1, 1, 1, 1], [-2, -1, -1, -1, -1, -1])

# invalid data_format, data_format is not same as Fmap's format
dynamic_invalid_format_case = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 data_format='NCDHW')

# invalid fmap shape d_dim = -2
dynamic_invalid_fmap_d_case = _run_api((1, -2, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid fmap shape h_dim = 0
dynamic_invalid_fmap_h_case = _run_api((1, -1, 0, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid fmap shape
dynamic_invalid_fmap_dims_case = _run_api((1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1])

# invalid dilations
dynamic_invalid_dilations_case = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 dilations=(2, 1, 1, 1, 1))

# invalid groups
dynamic_invalid_groups_case = _run_api((1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (21, 53), (1, 64), (20, 20), (256,256)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (1, -1, -1, -1, 256), "NDHWC", "float16",
                 [(1, 1), (26, 38), (1, 152), (10, 10), (256, 256)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 groups=2)

# Test another success case
dynamic_shape_2_case = _run_api((-2,), "NDHWC", "float16",
                 [(1, None), (1, None), (1, None), (1, None), (1, None)],
                 (3, 3, 3, 256, 256), 'DHWCN', 'float32',
                 [(3, 3), (3, 3), (3, 3), (256, 256), (256,256)],
                 (-2,), "NDHWC", "float16",
                 [(1, None), (1, None), (1, None), (1, None), (1, None)],
                 [1, 2, 2, 2, 1], [-1, -1, -1, -1, -1, -1],
                 groups=1)

# test pad_d < filter_d constraint
invalid_pads = [4, 4, 0, 0, 0, 0]
dynamic_invalid_pad_d_case = _run_api_v2(pads=invalid_pads)

# test pad_h < filter_h constraint
invalid_pads = [0, 0, 0, 4, 0, 0]
dynamic_invalid_pad_h_case = _run_api_v2(pads=invalid_pads)

# test pad_w < filter_w constraint
invalid_pads = [0, 0, 0, 0, 4, 4]
dynamic_invalid_pad_w_case = _run_api_v2(pads=invalid_pads)

# test convolution result Failed in D direction
x = {'ori_shape': (1, 3, -1, -1, 256), 'shape': (1, 3, -1, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (3, 3), (1, 64), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, 2, -1, -1, 256), 'shape': (1, 2, -1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (2, 2), (1, 152), (10, 10), (256, 256)]}
dynamic_d_res_not_match_case = _run_api_v2(x=x, out_backprop=out_backprop)

# test convolution result Failed in H direction
x = {'ori_shape': (1, -1, 3, -1, 256), 'shape': (1, -1, 3, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (3, 3), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, -1, 2, -1, 256), 'shape': (1, -1, 2, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (2, 2), (10, 10), (256, 256)]}
dynamic_h_res_not_match_case = _run_api_v2(x=x, out_backprop=out_backprop)

# test convolution result Failed in W direction
x = {'ori_shape': (1, -1, -1, 4, 256), 'shape': (1, -1, -1, 4, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (1, 64), (4, 4), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 5, 256), 'shape': (1, -1, -1, 5, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (1, 152), (5, 5), (256, 256)]}
dynamic_w_res_not_match_case = _run_api_v2(x=x, out_backprop=out_backprop)

# test Chip Design demand dedy_w must >=2 when dedy_h != 1
x = {'ori_shape': (1, -1, -1, 3, 256), 'shape': (1, -1, -1, 3, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (1, 64), (3, 3), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 2, 256), 'shape': (1, -1, -1, 2, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (70, 152), (2, 2), (256, 256)]}
dynamic_invalid_chip_demand_case = _run_api_v2(x=x, out_backprop=out_backprop)

# test_fail fmap h_lower too small
x = {'ori_shape': (1, -1, 2, -1, 256), 'shape': (1, -1, 2, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (2, 2), (20, 20), (256,256)]}
out_backprop = {'ori_shape': (1, -1, -1, 2, 256), 'shape': (1, -1, -1, 2, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (70, 152), (2, 2), (256, 256)]}
fmap_h_lower_too_small_case = _run_api_v2(x=x, out_backprop=out_backprop)

# test h and w range correction for Fmap
x = {'ori_shape': (1, -1, 3, -1, 256), 'shape': (1, -1, 3, -1, 256),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 1), (2, 10), (3, 3), (2, 10), (256,256)]}
out_backprop = {'ori_shape': (1, -1, 1, -1, 256), 'shape': (1, -1, 1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 1), (26, 38), (3, 3), (10, 10), (256, 256)]}
h_w_range_correction_case = _run_api_v2(x=x, out_backprop=out_backprop)

# NDC1HWC0 and fmap_n != dedy_n
x = {'ori_shape': (1, -1, -1, -1, 256), 'shape': (1, -1, 16, -1, -1, 16),
     'ori_format': "NDHWC", 'format': "NDC1HWC0", 'dtype': "float16", 
     "range": [(1, 1), (21, 53), (16, 16), (1, 64), (20, 20), (16,16)]}
out_backprop = {'ori_shape': (2, -1, -1, -1, 256), 'shape': (2, -1, -1, -1, 256),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(2, 2), (26, 38), (1, 152), (10, 10), (256, 256)]}
dynamic_batch_not_match_case = _run_api_v2(x=x, out_backprop=out_backprop)

# Test bl1 Dynamic add additional rows
x = {'ori_shape': (-1, 512, 26, 28, 28), 'shape': (-1, 512, 26, 28, 28),
     'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
     "range": [(2, 63), (512, 512), (26, 26), (28, 28), (28, 28)]}
filter_m = {'ori_shape': (1, 1, 1, 512, 128), 'shape': (1, 1, 1, 512, 128),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, 128, 26, 28, 28), 'shape': (-1, 128, 26, 28, 28),
                'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
                "range": [(2, 11), (128, 128), (26, 26), (28, 28), (28, 28)]}

additional_no_row = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, data_format="NCDHW")

# one row
x = {'ori_shape': (-1, -1, 7, 7, 576), 'shape': (-1, -1, 7, 7, 576),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 2), (1, 44), (7, 7), (7, 7), (576, 576)]}
filter_m = {'ori_shape': (1, 1, 1, 576, 128), 'shape': (1, 1, 1, 576, 128),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, -1, 7, 7, 128), 'shape': (-1, -1, 7, 7, 128),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(2, 156), (1, 13), (7, 7), (7, 7), (128, 128)]}

additional_one_row = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, data_format="NDHWC")

# two rows
x = {'ori_shape': (1, 512, -1, 14, 14), 'shape': (1, 512, -1, 14, 14),
     'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
     "range": [(1, 1), (512, 512), (1, 8), (14, 14), (14, 14)]}
filter_m = {'ori_shape': (1, 1, 1, 512, 1024), 'shape': (1, 1, 1, 512, 1024),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (1, 1024, -1, 14, 14), 'shape': (1, 1024, -1, 14, 14),
                'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
                "range": [(1, 1), (1024, 1024), (8, 195), (14, 14), (14, 14)]}

additional_two_row = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, data_format="NCDHW")

# Test Dynamic Batch
x = {'ori_shape': (-1, 1024, 32, 7, 7), 'shape': (-1, 1024, 32, 7, 7),
     'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
     "range": [(2, 443), (1024, 1024), (32, 32), (7, 7), (7, 7)]}
filter_m = {'ori_shape': (1, 1, 1, 1024, 2048), 'shape': (1, 1, 1, 1024, 2048),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, 2048, 32, 7, 7), 'shape': (-1, 2048, 32, 7, 7),
                'ori_format': "NCDHW", 'format': "NCDHW", 'dtype': "float16", 
                "range": [(1, 12), (2048, 2048), (32, 32), (7, 7), (7, 7)]}

dynamic_batch_case = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, data_format="NCDHW")

# Test Dynamic Batch With BL1 shape
x = {'ori_shape': (-1, 32, -1, -1, 2), 'shape': (-1, 32, -1, -1, 2),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 67), (32, 32), (32, 68), (31, 208), (2, 2)]}
filter_m = {'ori_shape': (7, 7, 7, 2, 512), 'shape': (7, 7, 7, 2, 512),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, 26, -1, -1, 512), 'shape': (-1, 26, -1, -1, 512),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 454), (26, 26), (20, 57), (23, 504), (512, 512)]}

dynamic_batch_bl1_case = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m)

# Test Dynamic Batch With BL1 shape else branch
x = {'ori_shape': (-1, 16, 32, 32, 384), 'shape': (-1, 16, 32, 32, 384),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 64), (16, 16), (32, 32), (32, 32), (384, 384)]}
filter_m = {'ori_shape': (1, 1, 1, 384, 384), 'shape': (1, 1, 1, 384, 384),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, 8, 16, 16, 384), 'shape': (-1, 8, 16, 16, 384),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(1, 174), (8, 8), (16, 16), (16, 16), (384, 384)]}
strides = (1, 2, 2, 2, 1)
dynamic_batch_bl1_case2 = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, strides=strides)

# Test Dynamic Batch With BL1 shape (bl1_k < width_grads)
x = {'ori_shape': (-1, 64, 256, 384, 4), 'shape': (-1, 64, 256, 384, 4),
     'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
     "range": [(1, 59), (64, 64), (256, 256), (384, 384), (4, 4)]}
filter_m = {'ori_shape': (7, 7, 7, 4, 32), 'shape': (7, 7, 7, 4, 32),
            'ori_format': 'DHWCN', 'format': 'DHWCN', 'dtype': 'float32'}
out_backprop = {'ori_shape': (-1, 15, 50, 76, 32), 'shape': (-1, 15, 50, 76, 32),
                'ori_format': "NDHWC", 'format': "NDHWC", 'dtype': "float16", 
                "range": [(2, 7), (15, 15), (50, 50), (76, 76), (32, 32)]}
strides = (1, 4, 5, 5, 1)
bl1_k_smaller_than_width_grads = _run_api_v2(x=x, out_backprop=out_backprop, filter=filter_m, strides=strides)

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(success_case, "success", "dynamic_pass_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(success_case2, "success", "dynamic_SAME_strides_2_2_2_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_pads_case, RuntimeError, "dynamic_invalid_pads_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_pad_2_case, RuntimeError, "dynamic_invalid_pad_2_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_format_case, RuntimeError, "dynamic_invalid_format_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_fmap_d_case, RuntimeError, "dynamic_invalid_fmap_d_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_fmap_h_case, RuntimeError, "dynamic_invalid_fmap_h_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_fmap_dims_case, RuntimeError, "dynamic_invalid_fmap_dims_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_dilations_case, RuntimeError, "dynamic_invalid_dilations_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_groups_case, RuntimeError, "dynamic_invalid_groups_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_shape_2_case, "success", "dynamic_shape_2_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_pad_d_case, RuntimeError, "dynamic_invalid_pad_d_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_pad_h_case, RuntimeError, "dynamic_invalid_pad_h_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_pad_w_case, RuntimeError, "dynamic_invalid_pad_w_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_d_res_not_match_case, RuntimeError, "dynamic_d_res_not_match_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_h_res_not_match_case, RuntimeError, "dynamic_h_res_not_match_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_w_res_not_match_case, RuntimeError, "dynamic_w_res_not_match_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_invalid_chip_demand_case, RuntimeError, "dynamic_invalid_chip_demand_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(fmap_h_lower_too_small_case, RuntimeError, "fmap_h_lower_too_small_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(h_w_range_correction_case, "success", "h_w_range_correction_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_batch_not_match_case, RuntimeError, "dynamic_batch_not_match_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(additional_no_row, "success", "additional_no_row", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(additional_one_row, "success", "additional_one_row", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(additional_two_row, "success", "additional_two_row", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_batch_case, "success", "dynamic_batch_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_batch_bl1_case, "success", "dynamic_batch_bl1_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(dynamic_batch_bl1_case2, "success", "dynamic_batch_bl1_case2", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(bl1_k_smaller_than_width_grads,
                                "success", "bl1_k_smaller_than_width_grads", True))

if __name__ == '__main__':
    ut_case.run(["Ascend910A"])
