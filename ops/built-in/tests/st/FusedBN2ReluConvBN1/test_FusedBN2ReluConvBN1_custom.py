#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT('fused_bn2_reluv2_conv2d_bn1', None, None)

fmap_input = {"ori_shape": [256, 64, 56, 56],
                "ori_format": "NCHW",
                "dtype:": "float16"}
filters_input = {"ori_shape": [64, 64, 3, 3],
                    "ori_format": "NCHW",
                    "dtype:": "float16"}
channel_in = filters_input["ori_shape"][1]
channel_out = filters_input["ori_shape"][0]
sum_input = {"ori_shape": [channel_in],
                "format": "ND",
                "dtype:": "float32"}
square_sum_input = {"ori_shape": [channel_in],
                    "ori_format": "NCHW",
                    "dtype:": "float32"}
scale_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
offset_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
padding = [1, 1, 1, 1]
stride = [1, 1, 1, 1]
dilation = [1, 1, 1, 1]
epsilon = 1e-8
factor = 0.1
groups = 1

moving_mean_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
moving_variance_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
bias_input = None
moving_mean_cur_output = None
moving_variance_cur_output = None
mean_output = None
variance_out = None
relu_output = None
mask_output = None
mask_output = None
convolution_output = None
sum_output = None
square_sum_output = None
tiling = {'BL1': [12, 1], 'AL0': [14, 3], 'BL0': [3, 4], 'BLOCK_DIM': [1, 1, 1],
                  'BLOCK_INNER': 1}
casename = 'case1'
kernel_name="fused_bn2_reluv2_conv2d_bn1_testtest" + casename,
auto_tune_tiling=tiling


ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        fmap_input, sum_input, square_sum_input,
        scale_input, offset_input, moving_mean_pre_input,
        moving_variance_pre_input, filters_input, bias_input,
        moving_mean_cur_output, moving_variance_cur_output,
        mean_output, variance_out,
        relu_output, mask_output,
        convolution_output, sum_output, square_sum_output,
        factor, epsilon, stride, padding, dilation, groups],
        'expect': 'success',
        'case_name': 'test1'})

fmap_input = {"ori_shape": [256, 64, 56, 56],
                "ori_format": "NCHW",
                "dtype:": "float16"}
filters_input = {"ori_shape": [256, 64, 1, 1],
                    "ori_format": "NCHW",
                    "dtype:": "float16"}
channel_in = filters_input["ori_shape"][1]
channel_out = filters_input["ori_shape"][0]
sum_input = {"ori_shape": [channel_in],
                "format": "ND",
                "dtype:": "float32"}
square_sum_input = {"ori_shape": [channel_in],
                    "ori_format": "NCHW",
                    "dtype:": "float32"}
scale_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
offset_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
padding = [0, 0, 0, 0]
stride = [1, 1, 1, 1]
dilation = [1, 1, 1, 1]
epsilon = 1e-8
factor = 0.1
groups = 1
moving_mean_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
moving_variance_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        fmap_input, sum_input, square_sum_input,
        scale_input, offset_input, moving_mean_pre_input,
        moving_variance_pre_input, filters_input, bias_input,
        moving_mean_cur_output, moving_variance_cur_output,
        mean_output, variance_out,
        relu_output, mask_output,
        convolution_output, sum_output, square_sum_output,
        factor, epsilon, stride, padding, dilation, groups],
        'expect': RuntimeError,
        'case_name': 'test2'})

# case7 ok use
print("case7")
fmap_input = {"ori_shape": [256, 128, 56, 56],
                "ori_format": "NCHW",
                "dtype:": "float16"}
filters_input = {"ori_shape": [128, 128, 3, 3],
                    "ori_format": "NCHW",
                    "dtype:": "float16"}
channel_in = filters_input["ori_shape"][1]
channel_out = filters_input["ori_shape"][0]
sum_input = {"ori_shape": [channel_in],
                "format": "ND",
                "dtype:": "float32"}
square_sum_input = {"ori_shape": [channel_in],
                    "ori_format": "NCHW",
                    "dtype:": "float32"}
scale_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
offset_input = {"ori_shape": [channel_in],
                "ori_format": "ND",
                "dtype:": "float32"}
padding = [0, 1, 0, 1]
stride = [1, 1, 2, 2]
dilation = [1, 1, 1, 1]
epsilon = 1e-4
factor = 0.1
groups = 1
moving_mean_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
moving_variance_pre_input = {"ori_shape": [filters_input["ori_shape"][1]]}
ut_case.add_case(
    ['Ascend910A'],
    {'params': [
        fmap_input, sum_input, square_sum_input,
        scale_input, offset_input, moving_mean_pre_input,
        moving_variance_pre_input, filters_input, bias_input,
        moving_mean_cur_output, moving_variance_cur_output,
        mean_output, variance_out,
        relu_output, mask_output,
        convolution_output, sum_output, square_sum_output,
        factor, epsilon, stride, padding, dilation, groups],
        'expect': 'success',
        'case_name': 'test7'})

if __name__ == '__main__':
    ut_case.run('Ascend910')
    exit(0)
