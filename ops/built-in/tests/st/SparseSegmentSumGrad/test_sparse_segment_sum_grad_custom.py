import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.sparse_segment_sum_grad
    importlib.reload(sys.modules.get("impl.dynamic.sparse_segment_sum_grad"))


if __name__ == '__main__':
    reload_check_support()