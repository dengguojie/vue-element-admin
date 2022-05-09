#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Testcase Manager Failures
"""
# Standard Packages
from enum import auto, Enum


class TESTCASE_FAILURES(Enum):
    """
    TESTCASE FAIL REASONS
    """
    DYN_INPUT_DTYPES_INVALID = auto()
    OUTPUT_DTYPES_INVALID = auto()
    STC_INPUT_DTYPES_INVALID = auto()
    OTHER_PARAMS_INVALID = auto()
    CONST_INPUT_INVALID = auto()
    DYN_INPUT_INVALID = auto()
    STC_INPUT_INVALID = auto()
    DYN_ORI_INPUT_INVALID = auto()
    STC_ORI_INPUT_INVALID = auto()
    DYN_OUTPUT_INFERENCE_FAILED = auto()
    DYN_ORI_OUTPUT_INFERENCE_FAILED = auto()
    STC_OUTPUT_INFERENCE_FAILED = auto()
    STC_ORI_OUTPUT_INFERENCE_FAILED = auto()
    DYN_INPUT_RANGE_INFERENCE_FAILED = auto()
    DYN_OUTPUT_RANGE_INFERENCE_FAILED = auto()
    STC_SHAPE_OUT_OF_BOUND = auto()
    STC_TENSOR_GT_DYN_TENSOR = auto()
    DYN_STC_SHAPE_LEN = auto()
    DYN_STC_SHAPE_DIFF = auto()
    STC_INPUT_SHAPE_ZERO = auto()
    DYN_INPUT_SHAPE_ZERO = auto()
    STC_OUTPUT_SHAPE_ZERO = auto()
    DYN_OUTPUT_SHAPE_ZERO = auto()