from unittest.mock import patch
from unittest.mock import MagicMock

from impl.softmax_grad import op_select_format as op_select_format
from impl.dynamic.softmax_grad import op_select_format as dy_op_select_format


def test_op_select_format_1():
    x1 = {"shape": (4096, 4096), "dtype": "float16", "format": "ND", "ori_shape": (4096, 4096), "ori_format": "ND"}
    x2 = {"shape": (4096, 4096), "dtype": "float16", "format": "ND", "ori_shape": (4096, 4096), "ori_format": "ND"}
    output = {"shape": (4096, 4096), "dtype": "float32", "format": "ND", "ori_shape": (4096, 4096), "ori_format": "ND"}
    op_select_format(x1, x2, output, -1)

def test_op_select_format_2():
    x1 = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    x2 = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    output = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    op_select_format(x1, x2, output, -1)

def test_op_select_format_3():
    x1 = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    x2 = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    output = {"shape": (32, 32, 4096), "dtype": "float16", "format": "ND", "ori_shape": (32, 32, 4096), "ori_format": "ND"}
    dy_op_select_format(x1, x2, output, -1)