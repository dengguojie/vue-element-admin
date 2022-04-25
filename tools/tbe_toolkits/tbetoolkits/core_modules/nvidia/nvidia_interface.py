#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
NVIDIA Interface
"""
# Standard Packages
import ctypes


class NVIDIAInterface:
    def __init__(self):
        ctypes.CDLL("libnvidia-ml.so")
