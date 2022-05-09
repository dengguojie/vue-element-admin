#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
CSV Field types
"""
# Standard Packages
from enum import auto, Enum


class FIELD_TYPES(Enum):
    """
    Expected types for all columns
    """
    STRING = auto()  # This is a pure string
    SHAPELIKE_DYN = auto()  # This is a dynamic shape, which means it supports negative values
    SHAPELIKE_STC = auto()  # This is a static shape, which means it doesn't support negative values
    SHAPELIKE_DYN_EX = auto()  # This is a dynamic output shape, which means it supports inference repr
    SHAPELIKE_STC_EX = auto()  # This is a static output shape, which means it supports inference repr
    SHAPELIKE_FLOAT = auto()  # This is a float shape like object, which means it supports float dims
    SHAPELIKE_FLOAT_SIGNED = auto()  # This is a signed float shape like object, which means it supports negative
    STRING_CONTAINER = auto()  # This is a string container, which means it must be a tuple[str]
    INT = auto()  # This is an integer, which means it must be an int
    INT_CONTAINER = auto()
    RANGELIKE = auto()  # This is a range of shape, its definition is based on dynamic shape operator
    BOOL = auto()  # This is a pure bool
    DICT = auto()  # This is a pure dict
    FLOAT = auto()  # This is a pure float
    FREE_EVAL = auto()  # FREE Evaluation