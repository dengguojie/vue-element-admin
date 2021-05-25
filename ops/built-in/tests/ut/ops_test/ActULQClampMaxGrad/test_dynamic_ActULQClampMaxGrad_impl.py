from op_test_frame.ut import OpUT


ut_case = OpUT('ActULQClampMaxGrad', 'impl.dynamic.act_ulq_clamp_max_grad', 'act_ulq_clamp_max_grad')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 100), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 100), (128, 128)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': 'success',
        'case_name': 'test_act_ulq_clamp_max_grad_float16',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
