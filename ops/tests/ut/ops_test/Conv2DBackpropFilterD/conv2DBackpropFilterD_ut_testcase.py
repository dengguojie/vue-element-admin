# -*- coding: UTF-8 -*-
conv2d_bp_filter_op_testcase = [
  # dedy_dtype, w_dtype, dx_dtype, dedy_shape, w_shape, dx_shape, dedy_format,
  # w_format, dx_format, input_size, stride, padding, expect, dataflow
  ("float16", "float16", "float32", (1, 128, 7, 7), (1, 128, 7, 7), (128, 128, 1, 1),
    "NCHW", "NCHW", "NCHW", (128, 128, 1, 1), (1, 1, 1, 1), "VALID"),

  ("float16", "float16", "float32", (51, 3, 56, 162), (51, 74, 56, 162), (74, 3, 5, 1),
    "NCHW", "NCHW", "NCHW", (74, 3, 5, 1), (1, 1, 1, 1), 'SAME'),

  ("float16", "float16", "float32", (50, 3, 200, 79), (50, 68, 200, 40), (68, 3, 5, 3),
    "NCHW", "NCHW", "NCHW", (68, 3, 5, 3),(1, 1, 1, 2), 'SAME'),

  ("float16", "float16", "float32", (41, 2, 210, 40), (41, 122, 210, 40), (122, 2, 7, 1),
    "NCHW", "NCHW", "NCHW", (122, 2, 7, 1), (1, 1, 1, 1), 'SAME'),

  ("float16", "float16", "float32", (21, 3, 132, 93), (21, 79, 132, 93), (79, 3, 3, 5),
    "NCHW", "NCHW", "NCHW", (79, 3, 3, 5), (1, 1, 1, 1), 'SAME'),

  ("float16", "float16", "float32", (60, 2, 100, 56), (60, 80, 100, 56), (80, 2, 5, 5),
    "NCHW", "NCHW", "NCHW", (80, 2, 5, 5), (1, 1, 1, 1), 'SAME'),

  # ("float16", "float16", "float32", (60, 100, 56, 2), (60, 100, 56, 80), (80, 5, 5, 2),
  #   "NHWC", "NHWC", "NHWC", (80, 5, 5, 2), (1, 1, 1, 1), 'SAME'),

  # ("float16", "float16", "float32", (100, 56, 60, 2), (100, 56, 60, 80), (5, 5, 80, 2),
  #   "HWCN", "NHWC", "HWCN", (5, 5, 80, 2), (1, 1, 1, 1), 'SAME'),

  # ("float16", "float16", "float32", (60, 2, 100, 56), (60, 80, 100, 56), (80, 2, 5, 5),
  #   "NCHW", "NCHW", "NCCC", (80, 2, 5, 5), (1, 1, 1, 1), 'SAME', RuntimeError),

  # ("float16", "float16", "float32", (60, 2, 100, 56, 1), (60, 80, 100, 56), (80, 2, 5, 5),
  #   "NCHW", "NCHW", "NCHW", (80, 2, 5, 5), (1, 1, 1, 1), 'SAME', RuntimeError),

  # ("float16", "float16", "float32", (60, 2, 100, 56), (60, 80, 100, 56, 1), (80, 2, 5, 5),
  #   "NCHW", "NCHW", "NCHW", (80, 2, 5, 5), (1, 1, 1, 1), 'SAME', RuntimeError),

  # ("float16", "float16", "float32", (21, 3, 132, 93), (21, 79, 132, 93), (79, 3, 3, 5),
  #   "NCHW", "NCHW", "NCHW", (79, 3, 3, 5), (1, 1, 1, 1, 1), 'SAME', RuntimeError),

  # ("float16", "float16", "float32", (21, 3, 132, 93), (21, 79, 132, 93), (79, 3, 3, 5),
  #   "NCHW", "NCHW", "NCHW", (79, 3, 3, 5), (2, 2, 2, 2), 'SAME', RuntimeError),
]
