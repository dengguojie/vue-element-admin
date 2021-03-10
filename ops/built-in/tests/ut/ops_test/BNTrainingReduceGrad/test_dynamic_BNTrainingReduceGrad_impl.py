#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# from op_test_frame.common import precision_info
from op_test_frame.ut import OpUT
ut_case = OpUT("BnTrainingReduceGrad", "impl.dynamic.bn_training_reduce_grad", "bn_training_reduce_grad")

# pylint: disable=too-many-arguments
def gen_bn_training_reduce_grad_case(shape_grads, shape_x_norm, shape_diff_scale, shape_diff_offset, shape_scale,
                                     shape_batch_variance, range_x, range_others, dtype, dtype_others, case_name_val):
    return {
        "params": [{
            "shape": shape_grads,
            "ori_shape": shape_grads,
            "dtype": dtype,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_x
        }, {
            "shape": shape_x_norm,
            "ori_shape": shape_x_norm,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_x
        }, {
            "shape": shape_diff_scale,
            "ori_shape": shape_diff_scale,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }, {
            "shape": shape_diff_offset,
            "ori_shape": shape_diff_offset,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }, {
            "shape": shape_scale,
            "ori_shape": shape_scale,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }, {
            "shape": shape_batch_variance,
            "ori_shape": shape_batch_variance,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }, {
            "shape": shape_batch_variance,
            "ori_shape": shape_batch_variance,
            "dtype": dtype_others,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }, {
            "shape": shape_grads,
            "ori_shape": shape_grads,
            "dtype": dtype,
            "format": "NC1HWC0",
            "ori_format": "NC1HWC0",
            "range": range_others
        }],
        "case_name": case_name_val,
        "expect": "success",
        "format_expect": [],
        "support_expect": True
    }


case1 = gen_bn_training_reduce_grad_case(
    (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (1, -1, 1, 1, 16), (1, -1, 1, 1, 16), (1, -1, 1, 1, 16),
    (1, -1, 1, 1, 16), ((1, None), (1, None), (1, None), (1, None), (1, None)),
    ((1, 1), (1, None), (1, 1), (1, 1), (1, None)), "float16", "float32", "bn_training_reduce_grad_dynamic")

ut_case.add_case(["Ascend910A"], case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
