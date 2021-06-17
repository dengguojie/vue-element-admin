from op_test_frame.ut import OpUT


ut_case = OpUT('ActULQClampMinGrad', 'impl.dynamic.act_ulq_clamp_min_grad', 'act_ulq_clamp_min_grad')

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': 'success',
        'case_name': 'test_act_ulq_clamp_min_grad_float16',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, None)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, None)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, None)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': RuntimeError,
        'case_name': 'test_act_ulq_clamp_min_grad_float16_shape_range_none',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (1, 2), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1, 2), 'ori_format': 'ND'}],
        'expect': RuntimeError,
        'case_name': 'test_act_ulq_clamp_min_grad_float16_output_shape_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': RuntimeError,
        'case_name': 'test_act_ulq_clamp_min_grad_float16_clamp_min_mask_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': RuntimeError,
        'case_name': 'test_act_ulq_clamp_min_grad_float16_x_clamped_loss_type_error',
        'support_expect': True})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (-1, -1), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (128, 8), 'ori_format': 'ND', 'range': [(1, 200), (128, 128)]},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,), 'ori_format': 'ND'}],
        'expect': RuntimeError,
        'case_name': 'test_act_ulq_clamp_min_grad_float16_output_type_error',
        'support_expect': True})


if __name__ == '__main__':
    ut_case.run('Ascend910A')
