#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.batch_matmul import batch_matmul_generalization


def test_batch_matmul_generalization_range_check():
    input_x1_dynamic = {'ori_shape': [-1, -1, -1, -1], 'shape': (-1, -1, -1, -1), 'range': ((4, 7), (1, 3), (1, 3), (1, 3)), 'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_range': ((4, 7), (1, 3), (1, 3), (1, 3))}
    input_x2_dynamic = {"ori_shape": [-1, -1], "shape":  [-1, -1], "range": ((1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND", "ori_range": ((1,3), (1,3))}
    output_dynamic = {"ori_shape": (-1, -1, -1, -1), "shape": (-1, -1, -1, -1), "ori_range": ((4,7), (1,3), (1,3), (1, 3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": ((1, 48),)}
    batch_matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_z=output_dynamic,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul_generalization",
                                generalize_config={"mode": "keep_rank"})

def test_batch_matmul_generalization_not_valid():
    input_x1_dynamic = {'ori_shape': [-1, -1, -1, -1], 'shape': (-1, -1, -1, -1), 'range': ((4, 7), (1, 3), (1, 3), (1, 3)), 'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_range': ((4, 7), (1, 3), (1, 3), (1, 3))}
    input_x2_dynamic = {"ori_shape": [-1, -1], "shape":  [-1, -1], "range": ((1,3), (1,3)), "dtype": 'float32', "format": "ND", "ori_format" : "ND", "ori_range": ((1,3), (1,3))}
    output_dynamic = {"ori_shape": (-1, -1, -1, -1), "shape": (-1, -1, -1, -1), "ori_range": ((4,7), (1,3), (1,3), (1, 3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": ((1, 48),)}
    batch_matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_z=output_dynamic,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul_generalization",
                                generalize_config={"mode": "keep_rank"})


if __name__ == '__main__':
    test_batch_matmul_generalization_range_check()
    test_batch_matmul_generalization_not_valid()