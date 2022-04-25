#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Postprocessing funcs
"""
# Standard Packages
import numpy as np
from .registry import register_postprocessing
# Third-Party Packages

def _depthwise_conv2d_backprop_filter_mul(out):
    t = out.shape[0] // 256
    temp = np.zeros((t, 1, 16, 16))
    out = out.reshape((t, 1, 16, 16))
    for i in range(t):
        temp[i,0,:,:] = np.eye(16)
    out = np.multiply(temp, out).reshape(t*16*16).astype(out.dtype)
    return out

@register_postprocessing(["depthwise_conv2d_backprop_filter"])
def _depthwise_conv2d_backprop_filter(context: "tbetoolkits.UniversalTestcaseStructure"):
    if context.dyn_prof_result.output_bytes[0] == "DYN_OFF":
        dyn_out =  context.dyn_prof_result.output_bytes
    else:
        out = np.frombuffer(context.dyn_prof_result.output_bytes[0], context.output_dtypes[0])
        dyn_out = _depthwise_conv2d_backprop_filter_mul(out)
        dyn_out = (dyn_out.tobytes(),)
    if context.stc_prof_result.output_bytes[0] == "STC_OFF":
        stc_out =  context.stc_prof_result.output_bytes
    else:
        out = np.frombuffer(context.stc_prof_result.output_bytes[0], context.output_dtypes[0])
        stc_out = _depthwise_conv2d_backprop_filter_mul(out)
        stc_out = (stc_out.tobytes(),)
    return dyn_out, stc_out, context.cst_prof_result.output_bytes, context.bin_prof_result.output_bytes

