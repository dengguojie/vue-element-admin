import imp
from importlib import reload

def test_get_op_support_info():
    from impl.dynamic.fused_mul_apply_momentum import get_op_support_info
    get_op_support_info({"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (1,), "dtype": "float16", "format":"ND", "ori_format":"ND", "ori_shape":(1,)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)},
                        {"shape": (144, 16, 16, 16), "dtype": "float16","format":"ND", "ori_format":"ND", "ori_shape":(144, 16, 16, 16)}, True)

def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.fused_mul_apply_momentum
    importlib.reload(sys.modules.get("impl.dynamic.fused_mul_apply_momentum"))

if __name__ == '__main__':
    reload_check_support()
    test_get_op_support_info()