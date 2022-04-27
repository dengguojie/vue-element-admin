#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import tbe
import importlib
from te import platform as cce_conf
from impl.dynamic.bn_training_reduce_grad import bn_training_reduce_grad


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    importlib.reload(sys.modules.get("impl.dynamic.bn_training_reduce_grad"))

def test_bn_training_reduce_grad_001():
    '''
    test_bn_training_reduce_grad_001
    '''
    input_list = [{
        'ori_shape': (1, 256, 1, 1),
        'shape': (1, 256, 1, 1),
        'range': ((1,1),(256,256),(1,1),(1,1)),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (1, 256, 1, 1),
        'shape': (1, 256, 1, 1),
        'range': ((1,1),(256,256),(1,1),(1,1)),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (256,),
        'shape': (256,),
        'range': ((256,256),),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (256,),
        'shape': (256,),
        'range': ((256,256),),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (256,),
        'shape': (256,),
        'range': ((256,256),),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (256,),
        'shape': (256,),
        'range': ((256,256),),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (256,),
        'shape': (256,),
        'range': ((256,256),),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }, {
        'ori_shape': (1, 256, 1, 1),
        'shape': (1, 256, 1, 1),
        'range': ((1,1),(256,256),(1,1),(1,1)),
        'ori_format': 'NCHW',
        'format': 'NCHW',
        'dtype': 'float32'
    }]
    with tbe.common.context.op_context.OpContext("static"):
        bn_training_reduce_grad(*input_list)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend910A")
    reload_check_support()
    test_bn_training_reduce_grad_001()
    cce_conf.te_set_version(soc_version)