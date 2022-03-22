import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.spence
    importlib.reload(sys.modules.get("impl.dynamic.spence"))


if __name__ == '__main__':
    reload_check_support()