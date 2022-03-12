#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from impl.strided_slice_strides_larger_than_one import StridedSliceStridesLargerThanOne

# 'pylint: disable=protected-access
def test_get_last_strides_parameters():
    """
    test for _get_last_strides_parameters
    """
    strided_slice = StridedSliceStridesLargerThanOne([1, 3, 320, 640], "float16",
                                                     [0, 0, 0, 1], [1, 3, 320, 640], [1, 1, 1, 2],
                                                     "strided_slice_strides_larger_than_one")
    strided_slice.aicore_num = 1
    strided_slice.ub_size = 63488
    strided_slice.total_ub_length = 126976
    strided_slice._get_last_strides_parameters()

if __name__ == '__main__':
    test_get_last_strides_parameters()
