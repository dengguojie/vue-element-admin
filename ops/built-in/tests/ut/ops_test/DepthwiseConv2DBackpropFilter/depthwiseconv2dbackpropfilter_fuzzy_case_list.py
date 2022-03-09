fuzzy_test_case = [
    {"inputs": [{'shape': (1, 1, 8, 3051, 16), 'ori_shape': (1, 8, 3051, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
                {'shape': (1, 1, 1, 3040, 16), 'ori_shape': (1, 1, 3040, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'ori_shape': (8, 12, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
                (1, 1, 3, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC',
                'test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs": [{'shape': (16, 1, 16, 16, 16), 'ori_shape': (16, 3, 16, 16), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
                {'shape': (16, 1, 14, 12, 16), 'ori_shape': (16, 3, 14, 12), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'ori_shape': (3, 1, 3, 5), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
                (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
                'depthwise_conv2d_backprop_filter_fuzz_build_generalization_general', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs":[{'shape': (50, 1, 35, 2896, 16), 'ori_shape': (50, 2, 35, 2896), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'shape': (50, 1, 26, 2888, 16), 'ori_shape': (50, 2, 26, 2888), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
               {'ori_shape': (2, 1, 10, 10), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'depthwise_conv2d_backprop_filter_fuzz_build_generalization_range_max_fixed', {"mode": "keep_rank"}],
     "expect": "success"},
    {"inputs":[{'shape': (6, 1, 2, 2857, 16), 'ori_shape': (6, 2, 2857, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
              {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
              {'shape': (6, 1, 1, 2857, 16),'ori_shape': (6, 1, 2857, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0','dtype': 'float16'},
              {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
              (1, 2, 1, 1), (1, 1, 1, 1), (4, 5, 0, 0), 'NHWC',
              'test_depthwise_conv2d_backprop_filter_fuzz_build_w_range_max_fixed',
              {"mode": "keep_rank"}],
     "expect": "lower_limit"},
    {"inputs":[{'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(4097, 5000), (4097, 5000))},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(4097, 5000), (4097, 5000))},
               {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_filter_fuzz_build_low_range_exceed_4096',
               {"mode": "keep_rank"}],
     "expect": "lower_limit"},
     {"inputs":[{'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 5000), (3000, 5000))},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 5000), (3000, 5000))},
               {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_filter_fuzz_build_up_range_exceed_4096',
               {"mode": "keep_rank"}],
     "expect": "upper_limit"},
     {"inputs":[{'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 4095), (3000, 4095))},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 4095), (3000, 4095))},
               {'ori_shape': (20, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_filter_fuzz_build_low_range_exceed_L1',
               {"mode": "keep_rank"}],
     "expect": "lower_limit"},
     {"inputs":[{'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(2000, 4095), (2000, 4095))},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(2000, 4095), (2000, 4095))},
               {'ori_shape': (8, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_filter_fuzz_build_up_range_exceed_l1',
               {"mode": "keep_rank"}],
     "expect": "upper_limit"},
     {"inputs":[{'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(64, 128), (64,128))},
               {'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(64, 128), (64, 128))},
               {'ori_shape': (3,3, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_filter_fuzz_build_up_range_exceed_l1',
               {"mode": "keep_rank"}],
     "expect": "success"},
]