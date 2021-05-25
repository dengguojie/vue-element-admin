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


if __name__ == '__main__':
    ut_case.run('Ascend910A')
