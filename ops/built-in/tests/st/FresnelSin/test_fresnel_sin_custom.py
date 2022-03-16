import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.fresnel_sin
    importlib.reload(sys.modules.get("impl.dynamic.fresnel_sin"))

if __name__ == '__main__':
    reload_check_support()