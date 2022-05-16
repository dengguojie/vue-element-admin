import numpy as np
from op_test_frame.ut import OpUT
from impl.dynamic.dynamic_rnn_grad import dynamic_rnn_grad_generalization
ut_case = OpUT("DynamicRNNGrad", "impl.dynamic.dynamic_rnn_grad", "dynamic_rnn_grad")

def test_rnn_grad_generalization(test_arg):
    dynamic_rnn_grad_generalization(
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (32,64), "dtype": "float16", "ori_shape": (32,64), "ori_format": "FRACTAL_ZN_LSTM", "format": "FRACTAL_ZN_LSTM"},
                    {"shape": (4*16,), "dtype": "float16", "ori_shape": (4*16,), "ori_format": "ND", "format": "ND"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,4,16), "dtype": "float16", "ori_shape": (1,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,4,16), "dtype": "float16", "ori_shape": (1,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,4,16), "dtype": "float16", "ori_shape": (1,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (1,4,16), "dtype": "float16", "ori_shape": (1,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    None, None, None, None,
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"},
                    {"shape": (2,4,16), "dtype": "float16", "ori_shape": (2,4,16), "ori_format": "FRACTAL_NZ", "format": "FRACTAL_NZ"})

ut_case.add_cust_test_func(test_func=test_rnn_grad_generalization)