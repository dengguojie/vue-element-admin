import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.concat_d
    importlib.reload(sys.modules.get("impl.dynamic.concat_d"))

if __name__ == '__main__':
    reload_check_support()

