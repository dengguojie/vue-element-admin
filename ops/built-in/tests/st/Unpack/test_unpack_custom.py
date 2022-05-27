import imp


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.unpack
    import impl.unpack
    importlib.reload(sys.modules.get("impl.dynamic.unpack"))
    importlib.reload(sys.modules.get("impl.unpack"))

if __name__ == '__main__':
    reload_check_support()