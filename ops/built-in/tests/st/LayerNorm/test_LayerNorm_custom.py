import imp


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.layer_norm
    import impl.layer_norm
    importlib.reload(sys.modules.get("impl.dynamic.layer_norm"))
    importlib.reload(sys.modules.get("impl.layer_norm"))

if __name__ == '__main__':
    reload_check_support()
    
