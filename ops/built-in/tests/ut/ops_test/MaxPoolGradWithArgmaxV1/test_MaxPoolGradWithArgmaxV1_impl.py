#!/usr/bin/env python
# -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("max_pool_grad_with_argmax_v1", "impl.max_pool_grad_with_argmaxv1")


def gen_max_pool_grad_with_argmax_v1_case(x_dict, grad_dict, argmax_dict, y_dict, ksize, strides,
                                          pads, dtype, dilation, ceil_mode, kernel_name_val,
                                          expect, calc_expect_func=None):
    if calc_expect_func:
        return {"params": [x_dict, grad_dict, argmax_dict, y_dict, ksize, strides, pads, dtype, dilation, ceil_mode],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True,
                "calc_expect_func": calc_expect_func}
    else:
        return {"params": [x_dict, grad_dict, argmax_dict, y_dict, ksize, strides, pads, dtype, dilation, ceil_mode],
                "case_name": kernel_name_val,
                "expect": expect,
                "support_expect": True}


ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 38, 38, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 38, 38),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 9, 92, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 16, 9, 92),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 0, 0, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_1", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 39, 39, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 39, 39),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 9, 92, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 16, 9, 92),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 1, 1, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_2", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 39, 39, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 39, 39),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 9, 97, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (1, 16, 9, 97),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 77, 77, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 77, 77),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 1, 1, 1), 3, (1, 1, 1, 1), True,
                     "max_pool_grad_with_arxmax_v1_3", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (2560, 4, 32, 100, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (2560, 64, 32, 100),
                      "ori_format": "NCHW"},
                     {"shape": (2560, 4, 16, 50, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (2560, 64, 16, 50),
                      "ori_format": "NCHW"},
                     {"shape": (2560, 4, 4, 51, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (2560, 64, 4, 51),
                      "ori_format": "NCHW"},
                     {"shape": (2560, 4, 32, 100, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (2560, 64, 32, 100),
                      "ori_format": "NCHW"},
                     (1, 2, 2, 1), (1, 2, 2, 1), (1, 0, 0, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_4", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 32, 5, 5),
                      "ori_format": "NCHW"},
                     {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 32, 5, 5),
                      "ori_format": "NCHW"},
                     {"shape": (2, 2, 169, 3, 16), "dtype": "uint16", "format": "NC1HWC0", "ori_shape": (2, 32, 169, 3),
                      "ori_format": "NCHW"},
                     {"shape": (2, 2, 5, 5, 16), "dtype": "float16", "format": "NC1HWC0", "ori_shape": (2, 32, 5, 5),
                      "ori_format": "NCHW"},
                     (1, 13, 13, 1), (1, 1, 1, 1), (1, 6, 6, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_5", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (1, 4, 640, 640, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 64, 640, 640),
                      "ori_format": "NCHW"},
                     {"shape": (1, 4, 320, 320, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 64, 320, 320),
                      "ori_format": "NCHW"},
                     {"shape": (1, 4, 9, 6401, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (1, 64, 9, 6401),
                      "ori_format": "NCHW"},
                     {"shape": (1, 4, 640, 640, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 64, 640, 640),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 1, 1, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_6", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (192, 128, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (192, 128, 7, 7),
                      "ori_format": "NCHW"},
                     {"shape": (192, 128, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (192, 128, 1, 1),
                      "ori_format": "NCHW"},
                     {"shape": (192, 128, 49, 2, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (192, 128, 49, 2),
                      "ori_format": "NCHW"},
                     {"shape": (192, 128, 7, 7, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (192, 128, 7, 7),
                      "ori_format": "NCHW"},
                     (1, 7, 7, 1), (1, 7, 7, 1), (1, 0, 0, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_7", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (32, 40, 20, 20, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 640, 20, 20),
                      "ori_format": "NCHW"},
                     {"shape": (32, 40, 20, 20, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 640, 20, 20),
                      "ori_format": "NCHW"},
                     {"shape": (32, 40, 169, 26, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (32, 640, 169, 26),
                      "ori_format": "NCHW"},
                     {"shape": (32, 40, 20, 20, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 640, 20, 20),
                      "ori_format": "NCHW"},
                     (1, 13, 13, 1), (1, 1, 1, 1), (1, 6, 6, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_8", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (32, 4, 112, 112, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 112, 112, 56),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 56, 56, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 56, 56),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 9, 197, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 9, 197),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 112, 112, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 112, 112),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 1, 1, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_9", True))


ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (32, 4, 120, 1200, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 120, 1200, 56),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 60, 600, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 60, 600),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 9, 2251, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 9, 2251),
                      "ori_format": "NCHW"},
                     {"shape": (32, 4, 120, 1200, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (32, 64, 120, 1200),
                      "ori_format": "NCHW"},
                     (1, 3, 3, 1), (1, 2, 2, 1), (1, 1, 1, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_10", True))

ut_case.add_case("Ascend910A",
                 gen_max_pool_grad_with_argmax_v1_case(
                     {"shape": (1, 1, 128, 512, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 128, 512),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 43, 171, 16), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 43, 171),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 25, 461, 16), "dtype": "uint16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 25, 461),
                      "ori_format": "NCHW"},
                     {"shape": (1, 1, 16, 128, 512), "dtype": "float16", "format": "NC1HWC0",
                      "ori_shape": (1, 16, 128, 512),
                      "ori_format": "NCHW"},
                     (1, 5, 5, 1), (1, 3, 3, 1), (1, 2, 2, 1), 3, (1, 1, 1, 1), False,
                     "max_pool_grad_with_arxmax_v1_11", True))

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
