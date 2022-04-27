#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


from impl.dynamic.fill import check_supported


def test_check_supported():
    dims = {'ori_shape': (3072,), 'shape': (3072,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'int32'}
    value = {'ori_shape': (1,), 'shape': (1,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    y = {'ori_shape': (3072,), 'shape': (3072,), 'ori_format': 'ND', 'format': 'ND', 'dtype': 'float16'}
    res, _ = check_supported(dims, value, y)


if __name__ == '__main__':
    test_check_supported()
