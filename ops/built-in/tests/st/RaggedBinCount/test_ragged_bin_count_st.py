"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

test_ragged_bin_count_st
"""
import numpy as np

# 'pylint: disable = unused-argument


def calc_expect_func_1(splits, values, size, weights, output, binary_output=False, kernel_name="ragged_bin_count"):
    splits_shape = splits["shape"]
    splits_data = splits["value"]
    values_shape = values["shape"]
    values_data = values["value"]
    size_data = size["value"]
    weights_shape = weights["shape"]
    weights_dtype = weights["dtype"]
    weights_data = weights["value"]
    output_dtype = output["dtype"]
    binary_output_data = binary_output

    result = np.zeros((splits_shape[0] - 1, size_data[0]), dtype=output_dtype)

    if splits_data[0] != 0:
        print(kernel_name, ": Splits must start with 0, not with ",
              splits_data[0])
        return np.NAN

    if weights_dtype != output_dtype and output_dtype != np.float:
        return np.NAN

    weights_num = 0
    if len(weights_shape) == 1:
        weights_num = weights_shape[0]
    elif len(weights_shape) == 2:
        weights_num = weights_shape[0] * weights_shape[1]

    batch_idx = 0
    for idx in range(0, values_shape[0]):
        while idx >= splits_data[batch_idx]:
            batch_idx += 1

        values_bin = values_data[idx]
        if values_bin < 0:
            print(kernel_name,
                  ": Input values must be non-negative, but get ", values_bin)
            return np.NAN

        if values_bin < size_data[0]:
            if binary_output_data:
                result[batch_idx - 1][values_bin] = 1
            else:
                if weights_num == 0:
                    result[batch_idx - 1][values_bin] += 1
                else:
                    result[batch_idx - 1][values_bin] += weights_data[idx]

    return [result, ]
