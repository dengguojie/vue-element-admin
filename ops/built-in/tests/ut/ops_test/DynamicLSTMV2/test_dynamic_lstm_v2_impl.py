# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
import numpy as np


ut_case = OpUT("DynamicLSTMV2", "impl.dynamic_lstm_v2", "dynamic_lstm_v2")


t = 2
ut_case.add_case(
    'Ascend310', 
    {
        'params': [
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'ori_shape': (t, 1, 512), 'shape': (t, 32, 1, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_ZN_LSTM', 'ori_format': 'ND', 'ori_shape': (768, 1024), 'shape': (48, 64, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (1024, ), 'shape': (1024, ), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (t, 1), 'shape': (t, 1), 'param_type': 'input'},
            None, None, None, None, None, None, None,
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_h'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_c'},
            None,
            None
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'dynamic_lstm_v2_case0',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True,
        # 'calc_expect_func': calc_expect_func
    },
)

print('run case 2')
ut_case.add_case(
    'Ascend310', 
    {
        'params': [
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'ori_shape': (t, 1, 512), 'shape': (t, 32, 1, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_ZN_LSTM', 'ori_format': 'ND', 'ori_shape': (768, 1024), 'shape': (48, 64, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (1024, ), 'shape': (1024, ), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (t, 1), 'shape': (t, 1), 'param_type': 'input'},
            None, None, None, None, None, None, None,
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_h'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_c'},
            None,
            None
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'dynamic_lstm_v2_case1',
        'expect': 'success',
        'support_expect': True,
        # 'calc_expect_func': calc_expect_func
    },
)

t = 3
print('run case 3')
ut_case.add_case(
    'Ascend310', 
    {
        'params': [
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'ori_shape': (t, 1, 512), 'shape': (t, 32, 1, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_ZN_LSTM', 'ori_format': 'ND', 'ori_shape': (768, 1024), 'shape': (48, 64, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (1024, ), 'shape': (1024, ), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (t, 1), 'shape': (t, 1), 'param_type': 'input'}, 
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (64, 1, 16, 16), 'shape': (64, 1, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'input'},
            None, None, None, None,
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            256, False, True, 0.0 
        ],
        'addition_params': {'impl_mode': 'high_precision'},
        'case_name': 'dynamic_lstm_v2_case2',
        'expect': 'success',
        # 'calc_expect_func': calc_expect_func
    },
)

ut_case.add_case(
    'Ascend910A', 
    {
        'params': [
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'ND', 'ori_shape': (t, 1, 512), 'shape': (t, 32, 1, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'FRACTAL_ZN_LSTM', 'ori_format': 'ND', 'ori_shape': (1024, 1024), 'shape': (64, 64, 16, 16), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (1024, ), 'shape': (1024, ), 'param_type': 'input'},
            {'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_shape': (1, 1), 'shape': (1, 1), 'param_type': 'input'},
            None, None, None, None, None, None, None,
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_h'},
            {'dtype': 'float16', 'format': 'FRACTAL_NZ', 'ori_format': 'NCHW', 'ori_shape': (t, 1, 256), 'shape': (t, 1, 256), 'param_type': 'output_c'},
            None,
            None
        ],
        'addition_params': {'impl_mode': 'high_performance'},
        'case_name': 'dynamic_lstm_v2_failed',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True,
    },
)


if __name__ == '__main__':
    ut_case.run("Ascend310")
