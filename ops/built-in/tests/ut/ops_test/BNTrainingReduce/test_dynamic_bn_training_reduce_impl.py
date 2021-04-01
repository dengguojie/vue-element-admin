#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BNTrainingReduce", "impl.dynamic.bn_training_reduce", "bn_training_reduce")


# pylint: disable=too-many-arguments
def gen_dynamic_bn_reduce_case(shape_x, shape_sum, shape_square, range_x, dtype_val, format_in, case_name_val, expect):
    """
    :param shape_x:
    :param shape_sum:
    :param shape_square:
    :param range_x:
    :param dtype_val:
    :param format_in:
    :param case_name_val:
    :param expect:
    :return:
    """
    return {
        "params": [{
            "ori_shape": shape_x,
            "shape": shape_x,
            "ori_format": "NC1HWC0",
            "format": format_in,
            "dtype": dtype_val,
            "range": range_x,
            "value": shape_x,
            "run_shape": shape_x
        }, {
            "ori_shape": shape_sum,
            "shape": shape_sum,
            "ori_format": "NC1HWC0",
            "format": format_in,
            "dtype": dtype_val,
            "range": range_x,
            "value": shape_sum,
            "run_shape": shape_sum
        }, {
            "ori_shape": shape_square,
            "shape": shape_square,
            "ori_format": "NC1HWC0",
            "format": format_in,
            "dtype": dtype_val,
            "range": range_x,
            "value": shape_square,
            "run_shape": shape_square
        }],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True
    }


CASE_1 = gen_dynamic_bn_reduce_case((-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16),
                                    ((1, None), (1, None), (1, None), (1, None), (16, 16)), "float16", "NC1HWC0",
                                    "dynamic_bn_training_reduce_1", "success")
CASE_2 = gen_dynamic_bn_reduce_case((16, 10, 17, 17, 16), (1, 10, 1, 1, 16), (1, 10, 1, 1, 16),
                                    ((16, 16), (10, 10), (17, 17), (17, 17), (16, 16)), "float32", "NC1HWC0",
                                    "dynamic_bn_training_reduce_2", "success")
CASE_3 = gen_dynamic_bn_reduce_case((-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16),
                                    ((1, None), (1, None), (1, None), (1, None), (16, 16)), "float32", "NC1HWC0",
                                    "dynamic_bn_training_reduce_3", "success")
CASE_4 = gen_dynamic_bn_reduce_case((-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16),
                                    ((1, None), (1, None), (1, None), (1, None), (16, 16)), "float16", "NCHW",
                                    "dynamic_bn_training_reduce_4", "failed")
CASE_5 = gen_dynamic_bn_reduce_case((-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1), (-1, -1, -1, -1, -1),
                                    ((1, None), (1, None), (1, None), (1, None), (16, 16)), "float32", "NCHW",
                                    "dynamic_bn_training_reduce_5", "failed")
CASE_6 = gen_dynamic_bn_reduce_case((-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16), (-1, -1, -1, -1, 16),
                                    ((1, None), (1, None), (1, None), (1, None), (16, 16)), "int32", "NC1HWC0",
                                    "dynamic_bn_training_reduce_6", "failed")

ut_case.add_case("Ascend910A", CASE_1)
ut_case.add_case("Ascend910A", CASE_2)
ut_case.add_case("Ascend910A", CASE_3)
ut_case.add_case("Ascend910A", CASE_4)
ut_case.add_case("Ascend910A", CASE_5)
ut_case.add_case("Ascend910A", CASE_6)

# ut_case.add_precision_case("Ascend910A", CASE_1)
# ut_case.add_precision_case("Ascend910A", CASE_2)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    # ut_case.run(["Ascend910A"], simulator_mode="pv", simlator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
    exit(0)
