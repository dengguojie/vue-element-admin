#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from typing import Mapping
import te
import te.platform as tepf
from te import platform as cce_conf
from impl.one_hot_d import check_supported


def test_check_supported():
    old_soc_version = tepf.get_soc_spec(tepf.SOC_VERSION)
    old_aicore_type = tepf.get_soc_spec(tepf.AICORE_TYPE)
    tepf.te_set_version("Ascend710", "VectorCore")
    input_x = {"shape": (240, 21128), "dtype": "int32", "format": "NCHW", "ori_shape": (240, 21128),
               "ori_format": "NCHW"}
    input_on_val = {"shape": (1,), "dtype": "int32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    input_off_val = {"shape": (1,), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"}
    depth = 2
    axis = 1
    res, _ = check_supported(input_x, input_on_val, input_off_val, depth, axis)
    if res:
        raise Exception("shape of input_x(240, 21128) is in black list, should return False")


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_check_supported()
    cce_conf.te_set_version(soc_version)
