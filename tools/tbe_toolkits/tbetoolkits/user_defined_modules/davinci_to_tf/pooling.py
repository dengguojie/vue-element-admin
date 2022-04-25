# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""pooling series tensorflow operators like max_pool"""
# Third-party Packages
import tbetoolkits
from .dav2tf_registry import register_func


@register_func(["avg_pool", ])
def _arg_max_v2(context: "tbetoolkits.UniversalTestcaseStructure"):
    context.stc_ori_inputs = context.stc_ori_inputs[:1]
    return context
