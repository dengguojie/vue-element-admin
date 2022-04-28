#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''
from te.platform.cce_conf import te_set_version
from impl.balance_rois import balance_rois


def test_balance_rois_01():
    te_set_version("Ascend710")
    input_list = [
        {
            'shape': (1000, 5),
            'ori_shape': (1000, 5),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'float16'
        }, {
            'shape': (1000, 5),
            'ori_shape': (1000, 5),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'float16'
        }, {
            'shape': (1000,),
            'ori_shape': (1000,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, 'balance_rois']
    balance_rois(*input_list)

def test_balance_rois_02():
    te_set_version("Ascend710")
    input_list = [
        {
            'shape': (10000, 5),
            'ori_shape': (10000, 5),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'float16'
        }, {
            'shape': (10000, 5),
            'ori_shape': (10000, 5),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'float16'
        }, {
            'shape': (10000,),
            'ori_shape': (10000,),
            'ori_format': 'ND',
            'format': 'ND',
            'dtype': 'int32'
        }, 'balance_rois']
    balance_rois(*input_list)
 
if __name__ == "__main__":
    test_balance_rois_01()
    test_balance_rois_02()