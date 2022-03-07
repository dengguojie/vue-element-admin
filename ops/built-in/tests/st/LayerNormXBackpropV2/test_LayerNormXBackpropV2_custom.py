import imp


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.layer_norm_x_backprop_v2
    import impl.layer_norm_x_backprop_v2
    importlib.reload(sys.modules.get("impl.dynamic.layer_norm_x_backprop_v2"))
    importlib.reload(sys.modules.get("impl.layer_norm_x_backprop_v2"))

if __name__ == '__main__':
    reload_check_support()