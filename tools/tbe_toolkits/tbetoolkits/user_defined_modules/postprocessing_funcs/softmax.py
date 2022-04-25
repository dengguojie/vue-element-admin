#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Postprocessing funcs
"""
# Standard Packages
import itertools
from .registry import register_postprocessing
# Third-Party Packages
import tbetoolkits
import numpy as np


def nc1hwc0_set_pad(out, shape, ori_shape, ori_dformat):
    C0 = shape[-1]
    C = ori_shape[1] if ori_dformat == "NCHW" else ori_shape[-1]
    Pad = 1+(C-1) % C0
    C1 = (C+C0-1)//C0
    out[:,C1-1,:,:,Pad:] = 0
    return out


def nz_set_pad(out, ori_shape):
    m, n = ori_shape[-2:]
    out[..., n:, m] = 0
    return out


@register_postprocessing(["softmax_v2"])
def softmax_postprocess(context: "tbetoolkits.UniversalTestcaseStructure"):
    dformat = context.stc_input_formats[0]
    dtype = context.stc_input_dtypes[0]

    # extract raw byte stream
    dyn_raw = context.dyn_prof_result.output_bytes
    stc_raw = context.stc_prof_result.output_bytes
    cst_raw = context.cst_prof_result.output_bytes
    bin_raw = context.bin_prof_result.output_bytes

    # if ND format, do nothing
    if dformat not in ["NC1HWC0", "FRACTAL_NZ"]:
        return dyn_raw, stc_raw, cst_raw, bin_raw

    # init some basic information
    ori_shape = context.stc_ori_inputs[0]
    shape = context.stc_inputs[0]

    # convert byte stream to numpy data
    dyn_out = np.zeros(shape) if isinstance(dyn_raw[0], str) else np.frombuffer(dyn_raw[0], dtype).reshape(shape)
    stc_out = np.zeros(shape) if isinstance(stc_raw[0], str) else np.frombuffer(stc_raw[0], dtype).reshape(shape)
    cst_out = np.zeros(shape) if isinstance(cst_raw[0], str) else np.frombuffer(cst_raw[0], dtype).reshape(shape)
    bin_out = np.zeros(shape) if isinstance(bin_raw[0], str) else np.frombuffer(bin_raw[0], dtype).reshape(shape)

    if dformat == "NC1HWC0":
        ori_dformat = context.stc_input_ori_formats[0]
        dyn_out = nc1hwc0_set_pad(dyn_out, shape, ori_shape, ori_dformat)
        stc_out = nc1hwc0_set_pad(stc_out, shape, ori_shape, ori_dformat)
        cst_out = nc1hwc0_set_pad(cst_out, shape, ori_shape, ori_dformat)
        bin_out = nc1hwc0_set_pad(bin_out, shape, ori_shape, ori_dformat)
    elif dformat == "FRACTAL_NZ":
        dyn_out = nz_set_pad(dyn_out, ori_shape)
        stc_out = nz_set_pad(stc_out, ori_shape)
        cst_out = nz_set_pad(cst_out, ori_shape)
        bin_out = nz_set_pad(bin_out, ori_shape)

    # prepare return data
    dyn_out = dyn_raw if isinstance(dyn_raw[0], str) else (dyn_out.tobytes(),)
    stc_out = stc_raw if isinstance(stc_raw[0], str) else (stc_out.tobytes(),)
    cst_out = cst_raw if isinstance(cst_raw[0], str) else (cst_out.tobytes(),)
    bin_out = bin_raw if isinstance(bin_raw[0], str) else (bin_out.tobytes(),)

    return dyn_out, stc_out, cst_out, bin_out

