import imp
from importlib import reload
from impl.dynamic.bias_add_grad import get_op_support_info

def test_get_op_support_info():
    get_op_support_info({"shape": (1, 10, 10, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (10, 10, 10, 10, 10),"ori_format": "NCHW"}, #x
                        {"shape": (1, 10, 10, 1, 16, 16), "dtype": "float32", "format": "FRACTAL_Z", "ori_shape": (10, 10, 10, 10, 10),"ori_format": "NCHW"},"NCHW")

def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.bias_add_grad
    import impl.bias_add_grad
    importlib.reload(sys.modules.get("impl.dynamic.bias_add_grad"))
    importlib.reload(sys.modules.get("impl.bias_add_grad"))

if __name__ == '__main__':
    test_get_op_support_info()
    reload_check_support()
    