#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import te
import tbe
from te.platform.cce_conf import te_set_version
from impl.dynamic.fill_d import fill_d


def test_fill_d_001():
    '''
    for fill_d single op
    '''
    input_list = [{"shape": (-1,), "ori_shape": (1,), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32", "range":[[1, 10]]},
                    {"shape": (2, 2), "ori_shape": (2, 2), "format": "NHWC", "ori_format": "NHWC", 'dtype': "int32", "range": [[2, 2], [2, 2]]},
                    (2, 2)]

    with tbe.common.context.op_context.OpContext("dynamic"):
        fill_d(*input_list)

if __name__ == '__main__':
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    test_fill_d_001()
    te_set_version(soc_version)