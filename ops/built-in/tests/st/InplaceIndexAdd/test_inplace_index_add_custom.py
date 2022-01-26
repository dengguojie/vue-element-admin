# -*- coding: UTF-8 -*-

import te.platform as tepf
from te import platform as cce_conf
from impl.inplace_index_add import inplace_index_add

def test_inplace_index_add():
    old_soc_version = tepf.get_soc_spec(tepf.SOC_VERSION)
    old_aicore_type = tepf.get_soc_spec(tepf.AICORE_TYPE)
    tepf.te_set_version("Ascend710", "VectorCore")
    var = {"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"}
    axis_indices = {"shape": (3,), "dtype": "int32", "format": "ND", "ori_shape": (3,), "ori_format": "ND"}
    updates = {"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"}
    var_out = {"shape": (3, 3, 2), "dtype": "int32", "format": "ND", "ori_shape": (3, 3, 2), "ori_format": "ND"}
    axis = 1
    inplace_index_add(var, axis_indices, updates, var_out, axis)

if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_inplace_index_add()
    cce_conf.te_set_version(soc_version)
