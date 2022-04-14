#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from importlib import reload
"""
test_elu_import
"""

def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.elu
    importlib.reload(sys.modules.get("impl.dynamic.elu"))

if __name__ == '__main__':
    reload_check_support()
    