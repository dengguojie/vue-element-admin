#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from te.platform.cce_conf import te_set_version
from impl.dynamic.max_pool_with_argmaxv2 import max_pool_with_argmax_v2


def test_maxpoolwithargmaxv2_001():
    '''
    for maxpoolwithargmaxv2 single op
    '''
    input_list = [{"shape": (32,4,112,112,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,112,112,16),"ori_format": "NC1HWC0"},
                    {"shape": (32,4,56,56,16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (32,4,56,56,16),"ori_format": "NC1HWC0"},
                    {"shape": (32,4,9,197,16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (32,4,9,197,16),"ori_format": "NC1HWC0"},
                    [1, 1, 3, 3],
                    [1, 1, 2, 2],
                    [1, 1, 1, 1]]
    with tbe.common.context.op_context.OpContext("dynamic"):
        max_pool_with_argmax_v2(*input_list)

if __name__ == '__main__':
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    test_maxpoolwithargmaxv2_001()
    te_set_version(soc_version)