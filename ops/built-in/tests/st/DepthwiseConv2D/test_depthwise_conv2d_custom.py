#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
custom st testcase
'''

from impl.dynamic.depthwise_conv2d import depthwise_conv2d_generalization

def test_depthwise_fuzzbuild_generalization_01():
    input_list = [
        {"ori_shape": (2, 32, 56, 56),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        {"ori_shape": (2, 2, 32, 1),
         "ori_format": "HWCN",
         "dtype": "float16"
        },
        None,
        None,
        {"ori_shape": (2, 32, 28, 28),
         "ori_format": "NCHW",
         "dtype": "float16"
        },
        (1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0), "NCHW", 0, "depthwise_fuzz_generalization"
    ]
    depthwise_conv2d_generalization(*input_list)

if __name__ == "__main__":
    test_depthwise_fuzzbuild_generalization_01()
