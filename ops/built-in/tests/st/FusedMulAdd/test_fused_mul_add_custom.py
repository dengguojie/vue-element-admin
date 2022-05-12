#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import imp
import json
from importlib import reload
from te import platform as cce_conf
from impl.dynamic.fused_mul_add import op_select_format

def test_op_select_format():
    """
    test_op_select_format
    """
    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_1")


    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_2")

    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (16, 16), "dtype": "float16", "format": "ND", "ori_shape": (16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_3")

    op_select_format({"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1, 2), "dtype": "float16", "format": "ND", "ori_shape": (1, 2), "ori_format": "ND"},
                     {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4), "ori_format": "ND"},
                     {"shape": (1, 2, 4), "dtype": "float16", "format": "ND", "ori_shape": (1, 2, 4), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_4")

    op_select_format({"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 24, 24, 16, 16), "ori_format": "ND"},
                     {"shape": [], "dtype": "float16", "format": "ND", "ori_shape": [], "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 24, 24, 16, 16), "ori_format": "ND"},
                     "test_fused_mul_add_op_select_format_5")

def test_op_support_info():
    """
    test_op_support_info
    """
    from impl.dynamic.fused_mul_add import get_op_support_info
    res = get_op_support_info({"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     {"shape": [], "dtype": "float16", "format": "ND", "ori_shape": [], "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 16, 24, 24, 16, 16), "dtype": "float16", "format": "ND", "ori_shape": (4, 16, 384, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_1")
    
    res_2 = get_op_support_info({"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     {"shape": (4, 1, 1, 384), "dtype": "float16", "format": "ND", "ori_shape": (4, 1, 1, 384), "ori_format": "ND"},
                     "test_fused_mul_add_op_support_info_2")
    
    split_maps = json.loads(res).get("_op_slice_info").get("splitMaps")
    assert len(split_maps) == 1
    for item in split_maps:
        input_list = item.get("inputList")
        assert len(input_list) == 2
        idx = input_list[0].get("idx")
        assert idx == 0
        idx = input_list[1].get("idx")
        assert idx == 2
    

    
def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.fused_mul_add
    importlib.reload(sys.modules.get("impl.dynamic.fused_mul_add"))

if __name__ == '__main__':
    reload_check_support()
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Hi3796CV300CS")
    test_op_select_format()
    test_op_support_info()
    cce_conf.te_set_version(soc_version)

