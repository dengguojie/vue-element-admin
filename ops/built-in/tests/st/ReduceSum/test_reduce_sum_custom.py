#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import imp
from importlib import reload


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    import importlib
    import sys
    import impl.dynamic.reduce_sum
    importlib.reload(sys.modules.get("impl.dynamic.reduce_sum"))

if __name__ == '__main__':
    reload_check_support()