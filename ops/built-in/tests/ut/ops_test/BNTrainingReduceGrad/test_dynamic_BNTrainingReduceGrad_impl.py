#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("BNTrainingReduceGrad", "impl.dynamic.bn_training_reduce_grad", "bn_training_reduce_grad")


# pylint: disable=too-many-arguments
def gen_bn_training_reduce_grad_case(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                     shape_batch_variance, range_x, range_others, dtype, dtype_others, shape_format,
                                     case_name_val, op_imply_type):
    return {
        "params": [{
            "shape": shape_grads,
            "ori_shape": shape_grads,
            "dtype": dtype,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_x
        }, {
            "shape": shape_x_norm,
            "ori_shape": shape_x_norm,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_x
        }, {
            "shape": shape_diff_scale,
            "ori_shape": shape_diff_scale,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }, {
            "shape": shape_diff_offset,
            "ori_shape": shape_diff_offset,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }, {
            "shape": shape_scale,
            "ori_shape": shape_scale,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }, {
            "shape": shape_batch_variance,
            "ori_shape": shape_batch_variance,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }, {
            "shape": shape_batch_variance,
            "ori_shape": shape_batch_variance,
            "dtype": dtype_others,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }, {
            "shape": shape_grads,
            "ori_shape": shape_grads,
            "dtype": dtype,
            "format": shape_format,
            "ori_format": shape_format,
            "range": range_others
        }],
        "case_name": case_name_val,
        "expect": "success",
        "format_expect": [],
        "support_expect": True,
        "op_imply_type": op_imply_type
    }


case1 = gen_bn_training_reduce_grad_case((-1, -1, -1, -1), (-1, -1, -1, -1), (1, -1, 1, 1), (1, -1, 1, 1),
                                         (1, -1, 1, 1), (1, -1, 1, 1), ((1, None), (1, None), (1, None), (1, None)),
                                         ((1, 1), (1, None), (1, 1), (1, 1)), "float16", "float32", "NCHW",
                                         "bn_training_reduce_grad_dynamic_001", "dynamic")

case2 = gen_bn_training_reduce_grad_case(
    (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (1, 64, 1, 1, 16), (1, 64, 1, 1, 16), (1, 64, 1, 1, 16),
    (1, 64, 1, 1, 16), ((1, None), (1, None), (1, None), (1, None), (16, 16)),
    ((1, 1), (64, 64), (1, 1), (1, 1), (1, 16)), "float16", "float32", "NC1HWC0", "bn_training_reduce_grad_dynamic_002",
    "dynamic")

case3 = gen_bn_training_reduce_grad_case(
    (-1, -1, -1, -1, -1, 16), (-1, -1, -1, -1, -1, 16), (1, 1, 64, 1, 1, 16), (1, 1, 64, 1, 1, 16),
    (1, 1, 64, 1, 1, 16), (1, 1, 64, 1, 1, 16), ((1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)),
    ((1, 1), (1, 1), (64, 64), (1, 1), (1, 1), (1, 16)), "float16", "float32", "NDC1HWC0",
    "bn_training_reduce_grad_dynamic_003", "dynamic")

case4 = gen_bn_training_reduce_grad_case(
    (1, 2, 1, 1, 16), (1, 2, 1, 1, 16), (1, 2, 1, 1, 16), (1, 2, 1, 1, 16), (1, 2, 1, 1, 16), (1, 2, 1, 1, 16),
    ((1, None), (1, None), (1, None), (1, None), (16, 16)), ((1, 1), (2, 2), (1, 1), (1, 1), (1, 16)), "float16",
    "float32", "NC1HWC0", "bn_training_reduce_grad_dynamic_004", "static")

case5 = gen_bn_training_reduce_grad_case(
    (1, 1, 2, 1, 1, 16), (1, 1, 2, 1, 1, 16), (1, 1, 2, 1, 1, 16), (1, 1, 2, 1, 1, 16), (1, 1, 2, 1, 1, 16),
    (1, 1, 2, 1, 1, 16), ((1, None), (1, None), (1, None), (1, None), (1, None), (16, 16)),
    ((1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 16)), "float16", "float32", "NDC1HWC0",
    "bn_training_reduce_grad_dynamic_005", "static")

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
