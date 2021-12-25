import numpy as np
from functools import reduce

def test_get_ub_core_for_cov():
    from impl.dynamic.transpose import get_ub_size
    from impl.dynamic.transpose import get_core_num
    from impl.dynamic.transpose import _static_scenario_goto_old_version
    from impl.dynamic.transpose import check_supported
    get_ub_size()
    get_core_num()
    shape_hit = [1, 128, 128, 3]
    shape_miss = [2, 128, 128, 3]
    _static_scenario_goto_old_version(shape_hit,  2)
    _static_scenario_goto_old_version(shape_miss, 2)
    input_x = {'ori_shape': (1, 128, 128, 3), 'shape': (1, 128, 128, 3), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (4,), 'shape': (4,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (1, 3, 18, 128), 'shape': (1, 3, 128, 128), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}

    res, _ = check_supported(input_x, perm, output_y)
    if res:
        raise Exception("1, 128, 128, 3 is in black list, should return False")


def calc_expect_10(x, perm, y):
    test_get_ub_core_for_cov()
    print("x=", x)
    print("perm=", perm)
    print("y=", y)
    x_value = x.get("value")
    res = np.transpose(x_value, axes=(1, 0))
    print("res=", res)
    return res 

