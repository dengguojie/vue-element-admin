from op_test_frame.ut import OpUT


ut_case = OpUT('ActsULQ', 'impl.dynamic.acts_ulq', 'acts_ulq')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': 'success',
        'case_name': 'test_acts_ulq_float16',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        4],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_num_bits_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, -1, 5, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_shape_range_none',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1, 2), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1, 2), 'ori_format': 'ND', 'range': [(1, 1), (2, 2)]},
        {'shape': (1, 2), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1, 2), 'ori_format': 'ND', 'range': [(1, 1), (2, 2)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_clamp_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, 5), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_output_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, 5), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_clamp_min_mask_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, 5), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_clamp_max_mask_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, 5), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_x_clamped_loss_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_output_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_clamp_min_mask_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_clamp_max_mask_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND', 'range': [(1, 1)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, -1, -1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        True,
        8],
        'expect': RuntimeError,
        'case_name': 'test_acts_ulq_float16_x_clamped_loss_type_error',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
