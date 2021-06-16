from op_test_frame.ut import OpUT


ut_case = OpUT('WtsARQ', 'impl.dynamic.wts_arq', 'wts_arq')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        8,
        True],
        'expect': 'success',
        'case_name': 'test_wts_arq_float16',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        8,
        False],
        'expect': 'success',
        'case_name': 'test_wts_arq_float16_offset_flag_false',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        4,
        True],
        'expect': RuntimeError,
        'case_name': 'test_wts_arq_float16_num_bits_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, 5, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, 5, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (5, 5), (1, None)]},
        8,
        True],
        'expect': RuntimeError,
        'case_name': 'test_wts_arq_float16_shape_range_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, -1, 5], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (5, 5)]},
        8,
        True],
        'expect': RuntimeError,
        'case_name': 'test_wts_arq_float16_y_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100), (1, 1), (1, 1), (1, 1)]},
        {'shape': [-1, -1, -1, -1], 'dtype': 'float32', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        8,
        True],
        'expect': RuntimeError,
        'case_name': 'test_wts_arq_float16_y_type_error',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
