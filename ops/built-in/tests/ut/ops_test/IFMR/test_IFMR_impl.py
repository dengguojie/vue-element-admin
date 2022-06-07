#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from impl.ifmr import ifmr
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe

ut_case = OpUT('ifmr', None, None)

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
         'ori_shape': (512,), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        True],
        'expect': 'success',
        'case_name': 'test_ifmr_float16_with_offset_910A'})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
         'ori_shape': (512,), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        True],
        'expect': 'success',
        'case_name': 'test_ifmr_float32_with_offset_910A'})

ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        {'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
         'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
         'ori_shape': (512,), 'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
         'ori_format': 'ND'},
        0.999999,
        0.999999,
        [0.7, 1.3],
        0.01,
        False],
        'expect': 'success',
        'case_name': 'test_ifmr_float16_without_offset_910A'})

def test_ifmr_float16_with_offset_SD3403(test_args):
    set_current_compile_soc_info("SD3403")
    with tbe.common.context.op_context.OpContext("static"):
        ifmr({'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
            'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
            'ori_shape': (512,), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            0.999999,
            0.999999,
            [0.7, 1.3],
            0.01,
            True)
    set_current_compile_soc_info(test_args)
ut_case.add_cust_test_func(test_func=test_ifmr_float16_with_offset_SD3403)

def test_ifmr_float16_without_offset_SD3403(test_args):
    set_current_compile_soc_info("SD3403")
    with tbe.common.context.op_context.OpContext("static"):
        ifmr({'shape': (32, 3, 5, 5), 'dtype': 'float16', 'format': 'ND',
            'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
            'ori_shape': (512,), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            0.999999,
            0.999999,
            [0.7, 1.3],
            0.01,
            False)
    set_current_compile_soc_info(test_args)
ut_case.add_cust_test_func(test_func=test_ifmr_float16_without_offset_SD3403)

def test_ifmr_float32_with_offset_310P3(test_args):
    set_current_compile_soc_info("Ascend310P3")
    with tbe.common.context.op_context.OpContext("static"):
        ifmr({'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
            'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
            'ori_shape': (512,), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            0.999999,
            0.999999,
            [0.7, 1.3],
            0.01,
            True)
    set_current_compile_soc_info(test_args)
ut_case.add_cust_test_func(test_func=test_ifmr_float32_with_offset_310P3)

def test_ifmr_float32_with_offset_610(test_args):
    set_current_compile_soc_info("Ascend610")
    with tbe.common.context.op_context.OpContext("static"):
        ifmr({'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
            'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
            'ori_shape': (512,), 'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
            'ori_format': 'ND'},
            0.999999,
            0.999999,
            [0.7, 1.3],
            0.01,
            True)
    set_current_compile_soc_info(test_args)
ut_case.add_cust_test_func(test_func=test_ifmr_float32_with_offset_610)

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         False],
#         'expect': 'success',
#         'case_name': 'test_ifmr_float32_without_offset'})\

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (10000, 1), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (10000, 1), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (100,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (100,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_indivisible'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (64, 3, 224, 224), 'dtype': 'float16', 'format': 'ND',
#          'ori_shape': (64, 3, 224, 224), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float16', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_mass_data_float16'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (64, 3, 224, 224), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (64, 3, 224, 224), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_mass_data_float32'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (64, 3, 224, 224), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (64, 3, 224, 224), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (4096,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (4096,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_large_cumsum_num_1'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (64, 3, 224, 224), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (64, 3, 224, 224), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (6144,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (6144,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_large_cumsum_num_2'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (64, 3, 224, 224), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (64, 3, 224, 224), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (8192,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (8192,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_large_cumsum_num_3'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.001,
#         True],
#         'expect': 'success',
#         'case_name': 'test_ifmr_mass_steps'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (1024, 3, 1024, 1024), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (1024, 3, 1024, 1024), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_data_num_out'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (2,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (2,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_min_shape_error'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (2,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (2,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_max_shape_error'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512, 1), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512, 1), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_cumsum_shape_error'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (9216,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (9216,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_cumsum_num_out'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         -0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_negative_search_step'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [1.3, 0.7],
#         0.01,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_search_range_decrease'})

# ut_case.add_case(
#     ['Ascend910'],
#     {'params': [
#         {'shape': (32, 3, 5, 5), 'dtype': 'float32', 'format': 'ND',
#          'ori_shape': (32, 3, 5, 5), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
#          'ori_shape': (512,), 'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
#          'ori_format': 'ND'},
#         0.999999,
#         0.999999,
#         [0.7, 1.3],
#         0.0001,
#         True],
#         'expect': ValueError,
#         'case_name': 'test_ifmr_excessive_steps'})

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "SD3403", "Ascend310P3", "Ascend610"])
    exit(0)
