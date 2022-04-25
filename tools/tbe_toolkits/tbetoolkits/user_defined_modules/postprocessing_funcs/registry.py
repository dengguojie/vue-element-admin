#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Postprocessing funcs
"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

# Third-Party Packages

postprocessing_func = {
}


def register_postprocessing(operator_names: Sequence[str]):
    """Register postprocessing function"""

    def __inner_postprocessing_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in postprocessing_func:
                logging.warning("postprocessing function of %s has already been registered!" % operator_name)
            postprocessing_func[operator_name] = __wrapper
        return __wrapper

    return __inner_postprocessing_registry
