from op_test_frame.ut import OpUT


ut_case = OpUT('WtsARQ', 'impl.dynamic.wts_arq', 'wts_arq')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100)]},
        {'shape': (-1, 1, 1, 1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': [16, 1, 1, 1], 'ori_format': 'ND', 'range': [(1, 100)]},
        {'shape': [-1, -1, -1, -1], 'dtype': 'float16', 'format': 'NCHW', 'ori_shape': [16, 6, 5, 5], 'ori_format': 'NCHW', 'range': [(1, 100), (1, 100), (1, 100), (1, 100)]},
        8,
        True],
        'expect': 'success',
        'case_name': 'test_wts_arq_float16',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
