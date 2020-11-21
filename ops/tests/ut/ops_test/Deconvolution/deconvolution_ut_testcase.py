# -*- coding: UTF-8 -*-
deconvolution_ut_case = [
    # soc, dtype, dedy_shape, filter_shape, dedx_shape, padding, stride, dilution, bias_flag
    # fp16 -> fp16 opti
    # stride = 1 with bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
    ),
    # stride > 1 withnot bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 5, 5),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        False,
    ),
    # fp16 -> fp16  general
    # stride = 1, no padding bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 2, 2),
        (16, 16, 3, 3),
        (1, 16, 4, 4),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
    ),
    # stride > 1, no padding no bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 4, 4),
        (16, 16, 4, 4),
        (1, 16, 10, 10),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        False,
    ),
    # stride = 1, padding bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 3, 3),
        (1, 16, 3, 3),
        (1, 1, 1, 1),
        (1, 1),
        (1, 1),
        True,
    ),
    # stride > 1, padding no bias
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 4, 4),
        (1, 16, 6, 6),
        (1, 1, 1, 1),
        (2, 2),
        (1, 1),
        False,
    ),
    # int8 -> int32 opti
    # stride = 1 with bias
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (32, 16, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
    ),
    # stride > 1 without bias
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (32, 16, 1, 1),
        (1, 16, 5, 5),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        False,
    ),
    # int8 -> int32 general
    # stride = 1, no padding no bias
    (
        ["Ascend310"],
        "int8",
        (1, 32, 2, 2),
        (32, 16, 3, 3),
        (1, 16, 4, 4),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        False,
    ),
    # stride > 1, no padding bias
    (
        ["Ascend310"],
        "int8",
        (1, 32, 4, 4),
        (32, 32, 4, 4),
        (1, 32, 10, 10),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        True,
    ),
    # stride = 1, padding no bias
    (
        ["Ascend310"],
        "int8",
        (2, 64, 3, 3),
        (64, 16, 3, 3),
        (2, 16, 3, 3),
        (1, 1, 1, 1),
        (1, 1),
        (1, 1),
        False,
    ),
    # stride > 1, padding bias
    (
        ["Ascend310"],
        "int8",
        (1, 64, 3, 3),
        (64, 16, 2, 2),
        (1, 16, 6, 6),
        (1, 1, 1, 1),
        (3, 3),
        (1, 1),
        True,
    ),
    # error case
    # input not math output
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 4, 4),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        RuntimeError,
    ),
    # error pading
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 4, 4),
        (0, 0),
        (1, 1),
        (1, 1),
        True,
        RuntimeError,
    ),
    # error shape
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 8, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 4, 4),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        RuntimeError,
    ),
    # error stride
    (
        ["Ascend310"],
        "int8",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (-1, 0),
        (1, 1),
        False,
        RuntimeError,
    ),
]


deconvolution_ut_fusion_case = [
    # soc dtype, dedy_shape, filter_shape, dedx_shape, padding, stride, dilution, bias_flag, fusionpass
    # the deconv+relu
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        False,
        ["relu"],
    ),
    (
        ["Ascend310", "Ascend710", "Ascend910"],
        "float16",
        (1, 16, 3, 3),
        (16, 16, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        ["relu"],
    ),
    # deconv + dequant [dquant, sqrt, vector_mode, relu_mode]
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (16, 32, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        ["dequant", True, True, False],
    ),
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (16, 32, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        ["dequant", True, False, True],
    ),
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (16, 32, 1, 1),
        (1, 16, 3, 3),
        (0, 0, 0, 0),
        (1, 1),
        (1, 1),
        True,
        ["dequant", False, True, True],
    ),
    # deconv + dequant + quant [quant, sqrt, vector_mode, relu_mode, sqrt, scalar, offset]
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (32, 32, 3, 3),
        (1, 32, 3, 3),
        (1, 1, 1, 1),
        (1, 1),
        (1, 1),
        True,
        ["quant", False, False, False, False, 1.1, -7],
    ),
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (32, 32, 3, 3),
        (1, 32, 3, 3),
        (1, 1, 1, 1),
        (1, 1),
        (1, 1),
        True,
        ["quant", True, False, False, False, 1, -7],
    ),
    (
        ["Ascend310"],
        "int8",
        (1, 32, 3, 3),
        (32, 32, 3, 3),
        (1, 32, 3, 3),
        (1, 1, 1, 1),
        (1, 1),
        (1, 1),
        True,
        ["quant", True, True, False, True, 0.2, 0],
    ),
    # deconv + requant  [requant, vector_mode, relu_mode]
    (
        ["Ascend710"],
        "int8",
        (1, 32, 4, 4),
        (32, 32, 4, 4),
        (1, 32, 10, 10),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        True,
        ["requant", True, False],
    ),
    (
        ["Ascend710"],
        "int8",
        (1, 32, 4, 4),
        (32, 32, 4, 4),
        (1, 32, 10, 10),
        (0, 0, 0, 0),
        (2, 2),
        (1, 1),
        True,
        ["requant", False, True],
    ),
]
