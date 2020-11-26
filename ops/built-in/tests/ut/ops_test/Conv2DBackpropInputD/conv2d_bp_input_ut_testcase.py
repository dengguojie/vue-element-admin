# -*- coding: UTF-8 -*-
"""
the Conv2DBackpropInputD ut testcase
"""
conv2d_bp_input_op_testcase = [
    # w_dtype, dedy_dtype, dx_dtype, w_shape, dedy_shape, dx_shape, dedy_format,
    # w_format, dx_format, input_size, stride, padding, dilations, groups, expect, dataflow
    # base case
    (
        "float16",
        "float16",
        "float16",
        (32, 32, 3, 3),
        (16, 32, 2, 2),
        (16, 32, 5, 5),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 32, 5, 5),
        (1, 1, 2, 2),
        (0, 0, 0, 0),
    ),
    (
        "float16",
        "float16",
        "float16",
        (32, 32, 3, 3),
        (16, 32, 2, 2),
        (16, 32, 5, 5),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 32, 5, 4),
        (1, 1, 2, 2),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 1, 1),
        (1, 64, 5, 4),
        (1, 64, 14, 11),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 14, 11),
        (1, 1, 3, 3),
        "VALID",
    ),
    (
        "float16",
        "float16",
        "float16",
        (16, 16, 1, 1),
        (1, 16, 2, 4096),
        (1, 16, 2, 4096),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 16, 2, 4096),
        (1, 1, 1, 1),
        "VALID",
    ),
    (
        "float16",
        "float16",
        "float16",
        (16, 16, 1, 1),
        (1, 16, 2, 128),
        (1, 16, 8, 512),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 16, 8, 512),
        (1, 1, 4, 4),
        "VALID",
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 16, 1, 1),
        (4, 64, 32, 32),
        (4, 16, 32, 32),
        "NCHW",
        "NCHW",
        "NCHW",
        (4, 16, 32, 32),
        (1, 1, 1, 1),
        "SAME",
    ),
    (
        "int8",
        "int8",
        "int32",
        (16, 64, 1, 1),
        (4, 64, 32, 32),
        (4, 16, 32, 32),
        "NCHW",
        "NCHW",
        "NCHW",
        (4, 16, 32, 32),
        (1, 1, 1, 1),
        "SAME",
    ),
    # vailid
    (
        "float16",
        "float16",
        "float16",
        (16, 16, 1, 1),
        (1, 16, 16, 16),
        (1, 16, 16, 16),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 16, 16, 16),
        (1, 1, 1, 1),
        "SAME",
    ),
    (
        "float16",
        "float16",
        "float16",
        (1024, 158, 1, 217),
        (1, 1024, 1, 25737),
        (1, 158, 1, 77209),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 158, 1, 77209),
        (1, 1, 1, 3),
        "SAME",
        (1, 1, 1, 2),
    ),
    # invaild
    (
        "float16",
        "float16",
        "float16",
        (8, 2, 15, 15),
        (34, 8, 28, 32),
        (34, 2, 12, 16),
        "NCHW",
        "NCHW",
        "NCHW",
        (34, 2, 12, 16),
        (1, 1, 1, 1),
        (15, 15, 15, 15),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    # test_conv2d_backprop_input.py
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        "ss",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (2, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "ND",
        "NCHW",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "ND",
        "NCHW",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (71, 1, 7, 1),
        (3, 71, 34, 172),
        (3, 1, 74, 172),
        "NCHW",
        "NCHW",
        "ND",
        (3, 1, 74, 172),
        (1, 1, 2, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    # conv1d
    (
        "float16",
        "float16",
        "float16",
        (256, 256, 1, 1),
        (16, 256, 1, 496),
        (16, 256, 1, 496),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 256, 1, 496),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (256, 256, 1, 1),
        (16, 256, 1, 496),
        (16, 256, 1, 1488),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 256, 1, 1488),
        (1, 1, 1, 3),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (128, 128, 1, 1),
        (16, 128, 1, 32000),
        (16, 128, 1, 64000),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 1, 64000),
        (1, 1, 1, 2),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    # test_conv2d_backprop_input_opti.py
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 1, 1),
        "VALID",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 28, 28),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 2, 2),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 19, 19),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 3, 3),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 12, 12),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 5, 5),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 1024, 1024),
        (16, 128, 1024, 1024),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 1024, 1024),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 500, 512),
        (16, 128, 500, 512),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 500, 512),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 60, 60),
        (16, 128, 60, 60),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 60, 60),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (1, 512, 56, 56),
        (1, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 128, 56, 56),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (32, 512, 56, 56),
        (32, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (32, 128, 56, 56),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 128, 128),
        (16, 128, 128, 128),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 128, 128),
        (1, 1, 1, 1),
        "SAME",
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 1, 1),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1, 1),
        (16, 512, 56, 56),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 512, 56, 1, 1),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "SAMESS",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float32",
        "float32",
        "float32",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 4097, 56),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 4097),
        (16, 512, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 56, 56),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 256, 1),
        (16, 512, 356, 356),
        (16, 128, 356, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 356, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 256, 1),
        (16, 512, 356, 356),
        (16, 512, 356, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 512, 356, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 256),
        (16, 512, 356, 356),
        (16, 128, 356, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 356, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 256),
        (16, 512, 356, 356),
        (16, 128, 356, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 356, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 16, 16),
        (16, 512, 356, 356),
        (16, 128, 356, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 356, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 16, 16),
        (16, 512, 356, 356),
        (16, 128, 4097, 356),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 4097, 356),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 16, 16),
        (16, 512, 356, 356),
        (16, 128, 356, 4097),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 356, 4097),
        (1, 1, 1, 1),
        "VAILD",
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
]


conv2d_bp_input_fusion_testcase = [
    # w_dtype, dedy_dtype, dx_dtype, w_shape, dedy_shape, dx_shape, dedy_format,
    # w_format, dx_format, input_size, stride, padding, dilations, groups, expect, dataflow
    # base case}
    # test_conv2d_backprop_input_genral.py
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 4, 4),
        (1, 64, 4, 4),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        "success",
        "drelu",
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 4, 4),
        (1, 64, 4, 4),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 2, 2),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        "success",
        "drelu",
    ),
    (
        "int8",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float32",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "int8",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float32",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "int8",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float32",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        "success",
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8),
        (64, 64, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 4800, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 4800, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (10000, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (80, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (128, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 300, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 300),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 20, 20),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 9, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 9),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (16000, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 5000, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 5000, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 5000),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 5000),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (63, 64, 8, 8),
        (64, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 128, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 128, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 9, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 9, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (64, 64, 8, 8),
        (64, 64, 8, 9),
        "NCHW",
        "NCHW",
        "NCHW",
        (64, 64, 8, 9),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 2, 3),
        (1, 64, 3, 8),
        (1, 64, 128, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 128, 8),
        (1, 1, 64, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 2),
        (1, 64, 8, 3),
        (1, 64, 8, 128),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 128),
        (1, 1, 1, 64),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 2, 2),
        (1, 64, 3, 3),
        (1, 64, 128, 128),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 128, 128),
        (1, 1, 64, 64),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (1, 64, 8, 262),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 256, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (1, 64, 8, 262),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 1, 1),
        (256, 1, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (1, 64, 8, 262),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 256, 1, 1),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (1, 64, 262, 16),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 1, 1),
        (1, 1, 1, 255),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    (
        "float16",
        "float16",
        "float16",
        (64, 64, 3, 3),
        (1, 64, 8, 8),
        (1, 64, 8, 8),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 64, 8, 8),
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (1, 1, 1, 1),
        1,
        RuntimeError,
    ),
    # test_conv2d_backprop_input_opti.py
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 56, 56),
        (16, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 56, 56),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (128, 128, 1, 1),
        (1, 128, 16, 16),
        (1, 128, 32, 32),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 128, 32, 32),
        (1, 1, 2, 2),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (128, 128, 1, 1),
        (1, 128, 16, 16),
        (1, 128, 32, 32),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 128, 32, 32),
        (1, 1, 2, 2),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (1, 512, 56, 56),
        (1, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 128, 56, 56),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (1, 512, 56, 56),
        (512, 128, 56, 56),
        "NCHW",
        "NCHW",
        "NCHW",
        (1, 128, 56, 56),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
    (
        "float16",
        "float16",
        "float16",
        (512, 128, 1, 1),
        (16, 512, 128, 128),
        (16, 128, 128, 128),
        "NCHW",
        "NCHW",
        "NCHW",
        (16, 128, 128, 128),
        (1, 1, 1, 1),
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        1,
        "success",
        "default",
    ),
]
