import imp
from importlib import reload

def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.add_n
    importlib.reload(sys.modules.get("impl.dynamic.add_n"))

if __name__ == '__main__':
    reload_check_support()