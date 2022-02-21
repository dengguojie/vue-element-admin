#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from impl.layer_norm import op_select_format
import numpy as np
ut_case = OpUT("LayerNorm", None, None)


def test_import_lib(test_arg):
    import sys
    import importlib
    modulename = sys.modules.get("impl.layer_norm")
    modulename1 = sys.modules.get("impl.layer_norm_unify")
    importlib.reload(modulename)
    importlib.reload(modulename1)


case1 = {"params": [{"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),
                     "ori_format": "NCHW"},
                    {"shape": (1023,), "dtype": "float16", "format": "NCHW", "ori_shape": (1023,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1023),
                     "ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case2 = {"params": [{"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),
                     "ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (1024,), "dtype": "float16", "format": "NCHW", "ori_shape": (1024,),"ori_format": "NCHW"},
                    {"shape": (64,128,1024), "dtype": "float16", "format": "NCHW", "ori_shape": (64,128,1024),
                     "ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    {"shape": (1,), "dtype": "float16", "format": "NCHW", "ori_shape": (1,),"ori_format": "NCHW"},
                    1, 2],
         "case_name": "layer_norm_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case21 = {"params": [{"shape": (128, 1024), "dtype": "float16", "format": "ND", "ori_shape": (128, 1024),
                      "ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "float16", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    {"shape": (1024, ), "dtype": "float16", "format": "ND", "ori_shape": (1024, ),"ori_format": "ND"},
                    {"shape": (128, 1024), "dtype": "float16", "format": "ND", "ori_shape": (128, 1024),
                     "ori_format": "ND"},
                    {"shape": (128, 1), "dtype": "float16", "format": "ND", "ori_shape": (128, 1),"ori_format": "ND"},
                    {"shape": (128, 1), "dtype": "float16", "format": "ND", "ori_shape": (128, 1),"ori_format": "ND"},
                    -1, -1],
         "addition_params": {'impl_mode': 'keep_fp16'},
         "case_name": "layer_norm_21",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case3 = {"params": [{"shape": (11,2,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (11,2,1023),
                     "ori_format": "NCHW"},
                    {"shape": (1023,), "dtype": "float16", "format": "NCHW", "ori_shape": (1023,),"ori_format": "NCHW"},
                    {"shape": (1023,), "dtype": "float16", "format": "NCHW", "ori_shape": (1023,),"ori_format": "NCHW"},
                    {"shape": (11,2,1023), "dtype": "float16", "format": "NCHW", "ori_shape": (11,2,1023),
                     "ori_format": "NCHW"},
                    {"shape": (11,2,1), "dtype": "float16", "format": "NCHW", "ori_shape": (11,2,1),
                     "ori_format": "NCHW"},
                    {"shape": (11,2,1), "dtype": "float16", "format": "NCHW", "ori_shape": (11,2,1),
                     "ori_format": "NCHW"},
                    -1, -1],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case4 = {"params": [{"shape": [3,3,31425], "dtype": "float16", "format": "NCHW", "ori_shape": [3,3,31425],
                     "ori_format": "NCHW"},
                    {"shape": [31425,], "dtype": "float16", "format": "NCHW", "ori_shape": [31425,],
                     "ori_format": "NCHW"},
                    {"shape": [31425,], "dtype": "float16", "format": "NCHW", "ori_shape": [31425,],
                     "ori_format": "NCHW"},
                    {"shape": [3,3,31425], "dtype": "float16", "format": "NCHW", "ori_shape": [3,3,31425],
                     "ori_format": "NCHW"},
                    {"shape": [3,3,1], "dtype": "float16", "format": "NCHW", "ori_shape": [3,3,1],"ori_format": "NCHW"},
                    {"shape": [3,3,1], "dtype": "float16", "format": "NCHW", "ori_shape": [3,3,1],"ori_format": "NCHW"},
                    2, 2],
         "case_name": "layer_norm_1",
         "expect": RuntimeError,
         "format_expect": [],
         "support_expect": True}
case5 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    1, 1],
         "case_name": "layer_norm_3",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case51 = {"params": [{"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (12288, 1024),"ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (1024,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (64, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 1024),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    1, 1],
         "addition_params": {'impl_mode': 'keep_fp16'},
         "case_name": "layer_norm_31",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case52 = {"params": [{"shape": (50, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ","ori_shape": (12288, 800),
                      "ori_format": "ND"},
                    {"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (800,), "dtype": "float16", "format": "ND", "ori_shape": (1024,),"ori_format": "ND"},
                    {"shape": (50, 768, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (12288, 800),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    {"shape": (12288, 1), "dtype": "float16", "format": "ND", "ori_shape": (12288, 1),
                     "ori_format": "ND"},
                    1, 1],
         "addition_params": {'impl_mode': 'high_performance'},
         "case_name": "layer_norm_52",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}
case6 = {"params": [{"shape": (256, 48, 13, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (256, 197, 768),"ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (256, 48, 13, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (256, 197, 768),"ori_format": "ND"},
                    {"shape": (256, 197), "dtype": "float16", "format": "ND", "ori_shape": (256, 197),
                     "ori_format": "ND"},
                    {"shape": (256, 197), "dtype": "float16", "format": "ND", "ori_shape": (256, 197),
                     "ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case7 = {"params": [{"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                     "ori_format": "ND"},
                    {"shape": (62,), "dtype": "float16", "format": "ND", "ori_shape": (62,),"ori_format": "ND"},
                    {"shape": (62,), "dtype": "float16", "format": "ND", "ori_shape": (62,),"ori_format": "ND"},
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                     "ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float16", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float16", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_7",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case8 = {"params": [{"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 63),
                     "ori_format": "ND"},
                    {"shape": (63,), "dtype": "float16", "format": "ND", "ori_shape": (63,),"ori_format": "ND"},
                    {"shape": (63,), "dtype": "float16", "format": "ND", "ori_shape": (63,),"ori_format": "ND"},
                    {"shape": (4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 63),
                     "ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float16", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float16", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_8",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case9 = {"params": [{"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 63),
                     "ori_format": "ND"},
                    {"shape": (63,), "dtype": "float32", "format": "ND", "ori_shape": (63,),"ori_format": "ND"},
                    {"shape": (63,), "dtype": "float32", "format": "ND", "ori_shape": (63,),"ori_format": "ND"},
                    {"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 63),
                     "ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_9",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case10 = {"params": [{"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                      "ori_format": "ND"},
                    {"shape": (62,), "dtype": "float32", "format": "ND", "ori_shape": (62,),"ori_format": "ND"},
                    {"shape": (62,), "dtype": "float32", "format": "ND", "ori_shape": (62,),"ori_format": "ND"},
                    {"shape": (4, 2, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (32, 62),
                     "ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    {"shape": (32, 1), "dtype": "float32", "format": "ND", "ori_shape": (32, 1),"ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_10",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case11 = {"params": [{"shape": (4, 300, 257, 12), "dtype": "float16", "format": "ND", "ori_shape": (4, 300, 257, 12),
                      "ori_format": "ND"},
                    {"shape": (12,), "dtype": "float16", "format": "ND", "ori_shape": (12,),"ori_format": "ND"},
                    {"shape": (12,), "dtype": "float16", "format": "ND", "ori_shape": (12,),"ori_format": "ND"},
                    {"shape": (4, 300, 257, 12), "dtype": "float16", "format": "ND", "ori_shape": (4, 300, 257, 12),
                     "ori_format": "ND"},
                    {"shape": (4, 300, 257, 1), "dtype": "float16", "format": "ND", "ori_shape": (4, 300, 257, 1),
                     "ori_format": "ND"},
                    {"shape": (4, 300, 257, 1), "dtype": "float16", "format": "ND", "ori_shape": (4, 300, 257, 1),
                     "ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_11",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case12 = {"params": [{"shape": (8,512,4096), "dtype": "float16", "format": "ND", "ori_shape": (8,512,4096),
                      "ori_format": "ND"},
                    {"shape": (4096,), "dtype": "float16", "format": "ND", "ori_shape": (4096,),"ori_format": "ND"},
                    {"shape": (4096,), "dtype": "float16", "format": "ND", "ori_shape": (4096,),"ori_format": "ND"},
                    {"shape": (8,512,4096), "dtype": "float16", "format": "ND", "ori_shape": (8,512,4096),
                     "ori_format": "ND"},
                    {"shape": (8,512, 1), "dtype": "float16", "format": "ND", "ori_shape": (8,512, 1),
                     "ori_format": "ND"},
                    {"shape": (8,512, 1), "dtype": "float16", "format": "ND", "ori_shape": (8,512, 1),
                     "ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_12",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case13 = {"params": [{"shape": (64,50,768), "dtype": "float16", "format": "ND", "ori_shape": (64,50,768),
                      "ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (64,50,768), "dtype": "float16", "format": "ND", "ori_shape": (64,50,768),
                     "ori_format": "ND"},
                    {"shape": (64,50, 1), "dtype": "float16", "format": "ND", "ori_shape": (64,50, 1),
                     "ori_format": "ND"},
                    {"shape": (64,50, 1), "dtype": "float16", "format": "ND", "ori_shape": (64,50, 1),
                     "ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_13",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case14 = {"params": [{"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                      "ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (768,), "dtype": "float16", "format": "ND", "ori_shape": (768,),"ori_format": "ND"},
                    {"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                     "ori_format": "ND"},
                    {"shape": (50,8, 1), "dtype": "float16", "format": "ND", "ori_shape": (50,8, 1),"ori_format": "ND"},
                    {"shape": (50,8, 1), "dtype": "float16", "format": "ND", "ori_shape": (50,8, 1),"ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_14",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case15 = {"params": [{"shape": (2,16,76800), "dtype": "float16", "format": "ND", "ori_shape": (2,16,76800),
                      "ori_format": "ND"},
                    {"shape": (76800,), "dtype": "float16", "format": "ND", "ori_shape": (76800,),"ori_format": "ND"},
                    {"shape": (76800,), "dtype": "float16", "format": "ND", "ori_shape": (76800,),"ori_format": "ND"},
                    {"shape": (2,16,76800), "dtype": "float16", "format": "ND", "ori_shape": (2,16,76800),
                     "ori_format": "ND"},
                    {"shape": (2,16, 1), "dtype": "float16", "format": "ND", "ori_shape": (2,16, 1),"ori_format": "ND"},
                    {"shape": (2,16, 1), "dtype": "float16", "format": "ND", "ori_shape": (2,16, 1),"ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_15",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case16 = {"params": [{"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                      "ori_format": "ND"},
                    {"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                     "ori_format": "ND"},
                    {"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                     "ori_format": "ND"},
                    {"shape": (50,8,768), "dtype": "float16", "format": "ND", "ori_shape": (50,8,768),
                     "ori_format": "ND"},
                    {"shape": (50,8, 1), "dtype": "float16", "format": "ND", "ori_shape": (50,8, 1),"ori_format": "ND"},
                    {"shape": (50,8, 1), "dtype": "float16", "format": "ND", "ori_shape": (50,8, 1),"ori_format": "ND"},
                    2, 0],
         "case_name": "layer_norm_16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case17 = {"params": [{"shape": (35, 65, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1040, 560),
                      "ori_format": "ND"},
                    {"shape": (560,), "dtype": "float16", "format": "ND", "ori_shape": (560,),"ori_format": "ND"},
                    {"shape": (560,), "dtype": "float16", "format": "ND", "ori_shape": (560,),"ori_format": "ND"},
                    {"shape": (35, 65, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1040, 560),
                     "ori_format": "ND"},
                    {"shape": (1040, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),"ori_format": "ND"},
                    {"shape": (1040, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),"ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_17",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case18 = {"params": [{"shape": (4, 3, 9, 7, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (4, 3, 9, 48, 112), "ori_format": "ND"},
                    {"shape": (112,), "dtype": "float16", "format": "ND", "ori_shape": (112,),"ori_format": "ND"},
                    {"shape": (112,), "dtype": "float16", "format": "ND", "ori_shape": (112,),"ori_format": "ND"},
                    {"shape": (4, 3, 9, 7, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (4, 3, 9, 48, 112), "ori_format": "ND"},
                    {"shape": (4, 3, 9, 48, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),
                     "ori_format": "ND"},
                    {"shape": (4, 3, 9, 48, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),
                     "ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_18",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case19 = {"params": [{"shape": (3, 12, 10, 6, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                      "ori_shape": (3, 12, 10, 48, 96), "ori_format": "ND"},
                    {"shape": (96,), "dtype": "float16", "format": "ND", "ori_shape": (112,),"ori_format": "ND"},
                    {"shape": (96,), "dtype": "float16", "format": "ND", "ori_shape": (112,),"ori_format": "ND"},
                    {"shape": (3, 12, 10, 6, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (3, 12, 10, 48, 96), "ori_format": "ND"},
                    {"shape": (3, 12, 10, 48, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),
                     "ori_format": "ND"},
                    {"shape": (3, 12, 10, 48, 1), "dtype": "float16", "format": "ND", "ori_shape": (1040, 1),
                     "ori_format": "ND"},
                    -1, -1],
         "case_name": "layer_norm_19",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case20 = {"params": [{"shape": (21,63,48,7), "dtype": "float32", "format": "ND","ori_shape": (21,63,48,7),
                       "ori_format": "ND"},
                    {"shape": (7,), "dtype": "float32", "format": "ND", "ori_shape": (7,),"ori_format": "ND"},
                    {"shape": (7,), "dtype": "float32", "format": "ND", "ori_shape": (7,),"ori_format": "ND"},
                    {"shape": (21,63,48,7), "dtype": "float32", "format": "ND","ori_shape": (21,63,48,7),
                     "ori_format": "ND"},
                    {"shape": (21,63,1,1), "dtype": "float32", "format": "ND", "ori_shape": (21,63,1,1),
                     "ori_format": "ND"},
                    {"shape": (21,63,1,1), "dtype": "float32", "format": "ND", "ori_shape": (21,63,1,1),
                     "ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_20",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case22 = {"params": [{"shape": (87,48,8,54), "dtype": "float32", "format": "ND","ori_shape": (87,48,8,54),
                       "ori_format": "ND"},
                    {"shape": (54,), "dtype": "float32", "format": "ND", "ori_shape": (54,),"ori_format": "ND"},
                    {"shape": (54,), "dtype": "float32", "format": "ND", "ori_shape": (54,),"ori_format": "ND"},
                    {"shape": (87,48,8,54), "dtype": "float32", "format": "ND","ori_shape": (87,48,8,54),
                     "ori_format": "ND"},
                    {"shape": (87,48,1,1), "dtype": "float32", "format": "ND", "ori_shape": (87,48,1,1),
                     "ori_format": "ND"},
                    {"shape": (87,48,1,1), "dtype": "float32", "format": "ND", "ori_shape": (87,48,1,1),
                     "ori_format": "ND"},
                    2, -1],
         "case_name": "layer_norm_22",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend710", "Ascend910A"], case21)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], case51)
ut_case.add_case(["Ascend310", "Ascend910A"], case52)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend710", "Ascend910A"], case14)
ut_case.add_case(["Ascend710", "Ascend910A"], case15)
ut_case.add_case(["Ascend710", "Ascend910A"], case16)
ut_case.add_case(["Ascend710", "Ascend910A"], case17)
ut_case.add_case(["Ascend710", "Ascend910A"], case18)
ut_case.add_case(["Ascend710", "Ascend910A"], case19)
ut_case.add_case(["Ascend710", "Ascend910A"], case20)
ut_case.add_case(["Ascend710", "Ascend910A"], case22)

# pylint: disable=unused-argument
def calc_expect_func(x, gamma, beta, y, mean, variance, begin_norm_axis,begin_params_axis):
    input_xArr = x['value']
    input_gammaArr = gamma['value']
    input_betaArr = beta['value']
    dtype = x['dtype']
    shape_x = x['shape']
    index_list = [index for index, _ in enumerate(shape_x)]
    reduce_axis = tuple(index_list[begin_norm_axis:])
    if dtype == "float16":
        input_xArr = input_xArr.astype(np.float32)
        input_gammaArr = input_gammaArr.astype(np.float32)
        input_betaArr = input_betaArr.astype(np.float32)

    mean = np.mean(input_xArr, reduce_axis, np.float32, keepdims=True)
    variance = np.mean(np.square(input_xArr - mean), reduce_axis, np.float32, keepdims=True)
    epsilon = 1e-12
    normalize = (input_xArr - mean) / np.sqrt(variance + epsilon)
    res = input_gammaArr*normalize + input_betaArr
    if dtype == "float16":
        mean = mean.astype(np.float16)
        variance = variance.astype(np.float16)
        res = res.astype(np.float16)
    return res, mean, variance


ut_case.add_case("all", {"params": [
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "output"},
    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "output"},
    {"shape": (1,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "output"},
    0, 0
],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)
})
ut_case.add_case("all", {"params": [
    {"shape": (2, 2, 768), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 768), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "input"},
    {"shape": (2, 2, 768), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 768), "ori_format": "NCHW",
     "param_type": "output"},
    {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "output"},
    {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
     "param_type": "output"},
    1, 2
],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

def test_op_select_format(test_arg):
    input_x = {"shape": (0,), "dtype": "float32", "format": "NCHW", "ori_shape": (0,),
               "ori_format": "NCHW", "param_type": "input"}
    input_gamma = {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                   "param_type": "input"}
    input_beta = {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                  "param_type": "input"}
    output_y = {"shape": (0,), "dtype": "float32", "format": "NCHW", "ori_shape": (0,),
                "ori_format": "NCHW",
                "param_type": "output"}
    output_mean = {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                   "param_type": "output"}
    output_variance = {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                       "param_type": "output"}
    begin_norm_axis = 1
    begin_params_axis = 2
    try:
        op_select_format(input_x, input_gamma, input_beta, output_y, output_mean, output_variance,
                         begin_norm_axis, begin_params_axis)
        assert False
    except RuntimeError as e:
        pass

    input_x = {"shape": (2, 2, 0), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 0),
               "ori_format": "NCHW",
               "param_type": "input"}
    input_gamma = {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                   "param_type": "input"}
    input_beta = {"shape": (768,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                  "param_type": "input"}
    output_y = {"shape": (2, 2, 0), "dtype": "float32", "format": "NCHW", "ori_shape": (2, 2, 0),
                "ori_format": "NCHW",
                "param_type": "output"}
    output_mean = {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                   "param_type": "output"}
    output_variance = {"shape": (2,), "dtype": "float32", "format": "NCHW", "ori_shape": (768,), "ori_format": "NCHW",
                       "param_type": "output"}
    try:
        op_select_format(input_x, input_gamma, input_beta, output_y, output_mean, output_variance,
                         begin_norm_axis, begin_params_axis)
        assert False
    except RuntimeError as e:
        pass


ut_case.add_cust_test_func(test_func=test_op_select_format)
ut_case.add_cust_test_func(test_func=test_import_lib)

if __name__ == "__main__":
    ut_case.run(["Ascend910A", "Ascend710", "Ascend310"])
