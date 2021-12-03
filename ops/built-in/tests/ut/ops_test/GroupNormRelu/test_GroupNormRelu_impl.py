# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from impl.group_norm_relu import group_norm_relu
from te import platform as cce_conf
ut_case = OpUT("GroupNormRelu")


def init_case():
    case_all = []
    dtype_all = ["float16", "float32"]
    format_5d = "NC1HWC0"
    format_nd = "ND"
    groups_num = 32
    eps = 0.00001
    batch_num = 8
    c0_num = 16
    kernel_name = "GroupNormRelu"
    input_shape_all = (
        (32, 7), (32, 14), (32, 28), (32, 15), (32, 30), (32, 60),
        (64, 30), (64, 14), (128, 7), (128, 15),
        (4, 56), (4, 120), (8, 28), (8, 56), (8, 60), (8, 120),
        (16, 14), (16, 28), (16, 30), (16, 56), (16, 60), (16, 120), )
    for data_type in dtype_all:
        for c1, h in input_shape_all:
            input_shape = (batch_num, c1, h, h, c0_num)
            ori_shape = (batch_num, c1 * c0_num, h, h)
            weight_shape = (c1 * c0_num,)
            params = [
                {"shape": input_shape, "dtype": data_type, "format": format_5d,
                 "ori_shape": ori_shape, "ori_format": format_nd},
                {"shape": weight_shape, "dtype": data_type, "format": format_nd,
                 "ori_shape": weight_shape, "ori_format": format_nd},
                {"shape": weight_shape, "dtype": data_type, "format": format_nd,
                 "ori_shape": weight_shape, "ori_format": format_nd},
                {"shape": input_shape, "dtype": data_type, "format": format_5d,
                 "ori_shape": ori_shape, "ori_format": format_nd},
                groups_num, eps, kernel_name
            ]
            case_all.append(params)
    return case_all


def add_case(_):
    case_all = init_case()
    cce_conf.cce_conf.te_set_version("Ascend710")
    for case in case_all:
        group_norm_relu(*case)
    cce_conf.cce_conf.te_set_version("Ascend910A")


ut_case.add_cust_test_func("Ascend910A", add_case)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
