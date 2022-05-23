#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Third-Party Packages
import tensorflow as tf
import numpy as np


def calc_expect_func(x,y):

    data_x = x.get('value')
    res = np.arccos(data_x)
    return res