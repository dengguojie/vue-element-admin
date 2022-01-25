import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.space_to_depth
    import impl.space_to_depth
    importlib.reload(sys.modules.get("impl.dynamic.space_to_depth"))
    importlib.reload(sys.modules.get("impl.space_to_depth"))

if __name__ == '__main__':
    reload_check_support()
    