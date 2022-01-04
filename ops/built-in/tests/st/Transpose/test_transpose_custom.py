#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from tbe.dsl import auto_schedule
from te import tvm
from te import platform as cce_conf

def test_nd_2_nz_shape_mismatch():
    from impl.dynamic.transpose import check_supported
    input_x = {'ori_shape': (24, 128, 3072), 'shape': (24, 192, 8, 16, 16), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    perm = {'ori_shape': (5,), 'shape': (5,), 'ori_format': 'NCDHW', 'format': 'NCDHW', 'dtype': 'float16'}
    output_y = {'ori_shape': (), 'shape': (48, 192, 16, 16), 'ori_format': 'ND', 'format': 'FRACTAL_NZ', 'dtype': 'float16'}
    res, reason = check_supported(input_x, perm, output_y)
    print(reason)

def test_get_ub_core_for_cov():
    from impl.dynamic.transpose import _static_scenario_goto_old_version
    from impl.dynamic.transpose import check_supported
    from impl.dynamic.transpose import get_ub_size
    from impl.dynamic.transpose import get_core_num
    from impl.dynamic.transpose import Transpose 

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

    class mocker(object):
        def __init__(self):
            self.x_dtype="int8"
    m = mocker()
    Transpose._move_data_s8(m, "abc")

if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend310")
    test_get_ub_core_for_cov()
    test_nd_2_nz_shape_mismatch()
    cce_conf.te_set_version(soc_version)
