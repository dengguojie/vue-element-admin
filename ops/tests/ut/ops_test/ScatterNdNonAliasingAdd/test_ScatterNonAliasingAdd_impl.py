"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ScatterNonAliasingAdd ut case
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterNonAliasingAdd", None, None)


# ============ auto gen ["Ascend910"] test cases start ===============

def gen_scatter_nd_add_case(x_shape, indices_shape, updates_shape,
                            dtype_x, case_name_val, expect):
    return {
        "params":
            [
                {
                    "shape": x_shape,
                    "dtype": dtype_x,
                    "ori_shape": x_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": indices_shape,
                    "dtype": "int32",
                    "ori_shape": indices_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": updates_shape,
                    "dtype": dtype_x,
                    "ori_shape": updates_shape,
                    "ori_format": "ND",
                    "format": "ND"
                },
                {
                    "shape": x_shape,
                    "dtype": dtype_x,
                    "ori_shape": x_shape,
                    "ori_format": "ND",
                    "format": "ND"
                }
            ],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


ut_case.add_case("all",
                 gen_scatter_nd_add_case((33, 5), (33, 25, 1), (33, 25, 5),
                                         "float32", "valid_fp32", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 14, 16), (10, 2, 1), (10, 2, 14, 16),
                                         "float16", "valid_fp16", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((8, 427), (2, 3, 1), (2, 3, 427),
                                         "int32", "valid_int32", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 512, 7, 7), (128, 1), (128, 512, 7, 7),
                                         "float16", "valid_fp16_2", "success"))

ut_case.add_case("all",
                 gen_scatter_nd_add_case((128, 32), (128, 1), (128, 32),
                                         "float32", "valid_fp32_2", "success"))

# ============ auto gen ["Ascend910"] test cases end =================
def calc_expect_func(inputs, indices, updates, outpus):
    input_data = inputs["value"]
    indices_data = indices["value"]
    updates_data = updates["value"]

    output = array_ops.scatter_nd_non_aliasing_add(input_data, indices_data, updates_data,name="output")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_data = sess.run(output)
        return output_data

ut_case.add_precision_case("Ascend910", {
    'params': [{'shape': (33, 5), 'dtype': 'float32', 'ori_shape': (33, 5), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 25, 1), 'dtype': 'int32', 'ori_shape': (33, 25, 1), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 25, 5), 'dtype': 'float32', 'ori_shape': (33, 25, 5), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (33, 5), 'dtype': 'float32', 'ori_shape': (33, 5), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    'params': [{'shape': (128, 14, 16), 'dtype': 'float32', 'ori_shape': (128, 14, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (10, 2, 1), 'dtype': 'int32', 'ori_shape': (10, 2, 1), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (10, 2, 14, 16), 'dtype': 'float32', 'ori_shape': (10, 2, 14, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (128, 14, 16), 'dtype': 'float32', 'ori_shape': (128, 14, 16), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    'params': [{'shape': (8, 427), 'dtype': 'float32', 'ori_shape': (8, 427), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (2, 3, 1), 'dtype': 'int32', 'ori_shape': (2, 3, 1), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (2, 3, 427), 'dtype': 'float32', 'ori_shape': (2, 3, 427), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (8, 427), 'dtype': 'float32', 'ori_shape': (8, 427), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    'params': [{'shape': (128, 32), 'dtype': 'float32', 'ori_shape': (128, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (128, 1), 'dtype': 'int32', 'ori_shape': (128, 1), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (128, 32), 'dtype': 'float32', 'ori_shape': (128, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "input"},
               {'shape': (128, 32), 'dtype': 'float32', 'ori_shape': (128, 32), 'ori_format': 'ND', 'format': 'ND', "param_type": "output"}],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})
if __name__ == '__main__':
    ut_case.run("Ascend910")
    # ut_case.run()
    exit(0)
