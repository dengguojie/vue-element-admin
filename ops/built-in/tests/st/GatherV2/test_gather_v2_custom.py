from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.gather_v2
    importlib.reload(sys.modules.get("impl.dynamic.gather_v2"))

if __name__ == '__main__':
    reload_check_support()
