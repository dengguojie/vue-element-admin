fuzzy_test_case = [
    {"inputs": [{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
                {'ori_shape': (2, 12, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
                {'shape': (1, 1, 8, 2000, 16), 'ori_shape': (1, 8, 2000, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'shape': (1, 1, 8, 2000, 16), 'ori_shape': (1, 8, 2000, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NHWC',
                'test_depthwise_conv2d_backprop_input_fuzz_build_w_range_max_fixed', {"mode": "keep_rank"}],
     "expect": [[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32', 'const_value': None,
               'const_value_range': [(1, 1), (4, 15), [2000, 2000], [1, 1]]}, {'ori_shape': (2, 12, 1, 1),
               'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'}, {'shape': (1, 1, 8, 2000, 16),
               'ori_shape': [-1, -1, -1, 1], 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16',
               'ori_range': [[1, 1], (4, 15), (1024, 4096), (1, 1)]}, {'shape': (1, 1, 8, 2000, 16),
               'ori_shape': [-1, -1, -1, 1], 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'strides': (1, 1, 1, 1)}, {'pads': (0, 0, 0, 0)}, {'dilations': (1, 1, 1, 1)}, {'data_format': 'NHWC'}]]},
    {"inputs": [{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
                {'ori_shape': (3, 1, 3, 5), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
                {'shape': (16, 1, 14, 12, 16), 'ori_shape': (16, 3, 14, 12), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'shape': (16, 1, 16, 16, 16), 'ori_shape': (16, 3, 16, 16), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
                (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
                'depthwise_conv2d_backprop_input_fuzz_build_generalization_general', {"mode": "keep_rank"}],
     "expect": [[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32',
               'const_value': None, 'const_value_range': [(16, 31), [3, 3], (6, 17), [16, 16]]}, {'ori_shape':
               (3, 1, 3, 5), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'}, {'shape':
               (16, 1, 14, 12, 16), 'ori_shape': [-1, 3, -1, -1], 'ori_format': 'NCHW', 'format': 'NC1HWC0',
               'dtype': 'float16', 'ori_range': [[16, 31], (3, 3), (4, 15), (4, 15)]}, {'shape': (16, 1, 16, 16, 16),
                'ori_shape': [-1, 3, -1, -1], 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
                {'strides': (1, 1, 1, 1)}, {'pads': (0, 0, 0, 0)}, {'dilations': (1, 1, 1, 1)},
                 {'data_format': 'NCHW'}]]},
    {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (2, 1, 10, 10), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'shape': (50, 1, 26, 2887, 16), 'ori_shape': (50, 2, 26, 2887), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
               {'shape': (50, 1, 35, 2896, 16), 'ori_shape': (50, 2, 35, 2896), 'ori_format': 'NCHW', 'format': 'NC1HWC0', 'dtype': 'float16'},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'depthwise_conv2d_backprop_input_fuzz_build_generalization_range_max_fixed', {"mode": "keep_rank"}],
     "expect": [[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32',
               'const_value': None, 'const_value_range': [(32, 2147483647), [2, 2], (25, 40), (2896, 2896)]},
               {'ori_shape': (2, 1, 10, 10), 'ori_format': 'NCHW', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
                {'shape': (50, 1, 26, 2887, 16), 'ori_shape': [-1, 2, -1, -1], 'ori_format': 'NCHW', 'format':
                'NC1HWC0', 'dtype': 'float16', 'ori_range': [[32, 2147483647], (2, 2), (16, 31), (1024, 3116)]},
                 {'shape': (50, 1, 35, 2896, 16), 'ori_shape': [-1, 2, -1, -1], 'ori_format': 'NCHW', 'format':
                 'NC1HWC0', 'dtype': 'float16'}, {'strides': (1, 1, 1, 1)}, {'pads': (0, 0, 0, 0)}, {'dilations':
                 (1, 1, 1, 1)}, {'data_format': 'NCHW'}]]},
    {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
              {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
              {'shape': (6, 1, 1, 2857, 16),'ori_shape': (6, 1, 2857, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0','dtype': 'float16'},
              {'shape': (6, 1, 2, 2857, 16), 'ori_shape': (6, 2, 2857, 1), 'ori_format': 'NHWC', 'format': 'NC1HWC0', 'dtype': 'float16'},
              (1, 2, 1, 1), (1, 1, 1, 1), (4, 5, 0, 0), 'NHWC',
              'test_depthwise_conv2d_backprop_input_fuzz_build_w_range_max_fixed',
              {"mode": "keep_rank"}],
     "expect": [{'result': 'UNSUPPORTED'}]},
    {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(4097, 5000), (4097, 5000))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(4097, 5000), (4097, 5000))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_input_fuzz_build_low_range_exceed_4096',
               {"mode": "keep_rank"}],
     "expect": [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['lower_limit']}}]},
     {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (11, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 5000), (3000, 5000))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 5000), (3000, 5000))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_input_fuzz_build_up_range_exceed_4096',
               {"mode": "keep_rank"}],
     "expect": [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['upper_limit']}}]},
     {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (20, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 4095), (3000, 4095))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(3000, 4095), (3000, 4095))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_input_fuzz_build_low_range_exceed_L1',
               {"mode": "keep_rank"}],
     "expect": [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['lower_limit']}}]},
     {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (8, 1, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(2000, 4095), (2000, 4095))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1,1),(2000, 4095), (2000, 4095))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_input_fuzz_build_up_range_exceed_l1',
               {"mode": "keep_rank"}],
     "expect": [{'result': 'UNSUPPORTED', 'reason': {'param_index': [2], 'type': ['upper_limit']}}]},
     {"inputs":[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'},
               {'ori_shape': (3, 3, 1, 1), 'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(64, 128), (64, 128))},
               {'ori_shape': (-1, 1, -1, -1), 'ori_format': 'NCHW', 'dtype': 'float16', "ori_range": ((32, 4096), (1, 1),(64, 128), (64, 128))},
               (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), 'NCHW',
               'test_depthwise_conv2d_backprop_input_fuzz_build_range_ok',
               {"mode": "keep_rank"}],
     "expect": [[{'shape': (4,), 'ori_shape': (4,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'}, {'ori_shape': (3, 3, 1, 1),
               'ori_format': 'HWCN', 'format': 'FRACTAL_Z', 'dtype': 'float16'}, {'ori_shape': [-1, 1, -1, -1],
               'ori_format': 'NCHW', 'dtype': 'float16', 'ori_range': ((32, 4096), (1, 1), (64, 128), (64, 128))},
                {'ori_shape': [-1, 1, -1, -1], 'ori_format': 'NCHW', 'dtype': 'float16', 'ori_range': ((32, 4096),
                (1, 1), (64, 128), (64, 128))}, {'strides': (1, 1, 1, 1)}, {'pads': (0, 0, 0, 0)}, {'dilations':
                (1, 1, 1, 1)}, {'data_format': 'NCHW'}]]},
]