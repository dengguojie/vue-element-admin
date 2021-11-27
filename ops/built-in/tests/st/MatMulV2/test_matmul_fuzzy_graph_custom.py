#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.dynamic.mat_mul import matmul_generalization


def test_matmul_generalization_upper_bound_input1():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, None)], 'ori_range': [(1, 48), (16369, None)]}
    input_x2_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    output_dynamic ={'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, None)], 'ori_range': [(1, 48), (16369, None)]}
    bias_dynamic = None

    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})

def test_matmul_generalization_unknown_rank():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, 2147483647)], 'ori_range': [(1, 48), (16369, 2147483647)]}
    input_x2_dynamic = {'ori_shape': [-2], 'dtype': 'float16', 'shape': [-2], 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    output_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (16369, 2147483647)], 'ori_range': [(1, 48), (16369, 2147483647)]}
    bias_dynamic = None
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})

def test_matmul_generalization_lower_bound_input2():
    input_x1_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(1, 48), (1, 48)], 'ori_range': [(1, 48), (1, 48)]}
    input_x2_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(2147483677, 2147483677), (1, 48)], 'ori_range': [(2147483677, 2147483677), (1, 48)]}
    output_dynamic = {'ori_shape': (-1, -1), 'dtype': 'float16', 'shape': (-1, -1), 'format': 'ND', 'ori_format': 'ND', 'range': [(2147483677, 2147483677), (1, 48)], 'ori_range': [(2147483677, 2147483677), (1, 48)]}
    bias_dynamic = {'ori_shape': -1, 'dtype': 'float16', 'shape': -1, 'format': 'ND', 'ori_format': 'ND', 'range': ((2147483677, 2147483677),), 'ori_range': ((2147483677, 2147483677),)}
    matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_y=output_dynamic,
                          trans_a=False, trans_b=False, kernel_name="matmul_generalization",
                          generalize_config={"mode": "keep_rank"})


if __name__ == '__main__':
    test_matmul_generalization_upper_bound_input1()
    test_matmul_generalization_unknown_rank()
    test_matmul_generalization_lower_bound_input2()