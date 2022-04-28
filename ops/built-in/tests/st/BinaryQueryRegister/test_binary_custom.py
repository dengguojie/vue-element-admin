#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.binary_query_register
    importlib.reload(sys.modules.get("impl.dynamic.binary_query_register"))

if __name__ == '__main__':
    reload_check_support()
