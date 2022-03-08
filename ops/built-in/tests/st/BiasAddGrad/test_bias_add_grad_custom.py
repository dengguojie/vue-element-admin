import imp
from importlib import reload


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
    reload_check_support()
    