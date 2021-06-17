from op_test_frame.ut import OpUT


ut_case = OpUT('ActsULQInputGrad', 'impl.dynamic.acts_ulq_input_grad', 'acts_ulq_input_grad')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]}],
        'expect': 'success',
        'case_name': 'test_acts_ulq_input_grad_float16',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]}],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_input_grad_float16_shape_range_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, 5), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]}],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_input_grad_float16_x_grad_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]}],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_input_grad_float16_clamp_min_mask_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]}],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_input_grad_float16_clamp_max_mask_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]}],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_input_grad_float16_x_grad_type_error',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
