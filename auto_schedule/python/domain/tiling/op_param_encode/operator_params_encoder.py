#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

TBE operator param encoder
"""
import json

class BaseClassParamsEncoder():
    """
    Base class for Params Encoder
    """

    def __init__(self):
        """
        init the support param dict
        """
        # Encode the type of tensor data
        self.dtype_dict = {
            'uint8': 0,
            'int8': 1,
            'float16': 2,
            'int32': 3,
            'float32': 4,
            'int16': 5
            }

        # Encode the type of op
        self.op_type_dict = {
            'conv2d': 0, \
            'conv2d_backprop_input': 1, \
            'conv2d_backprop_filter': 2, \
            'depthwise_conv2d_forward': 3,
            'depthwise_bp_input': 4, \
            'depthwise_bp_filter': 5, \
            'depthwise_conv2d_native_v200': 6, \
            'matmul': 7, \
            'convolution_3d': 8
            }

    def check_param_type(self, param, type_list):
        """check whether the type of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        type_list: type
            type of data
        """
        check_list = tuple(type_list)
        if not isinstance(param, check_list):
            raise TypeError("the type of param is error, \
                only support %s, but the type of param is %s" % \
                (str(type_list), type(param)))

    def check_param_length(self, param, length_list):
        """check whether the length of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        length_list: list
            length of data
        """
        if not len(param) in length_list:
            raise ValueError("the length of param is error, \
                only support %s, but the length of param is %s" % \
                (str(length_list), len(param)))

    def check_support_range(self, param, support_dict):
        """check whether the range of param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        support_dict: dict
            support context of data
        """
        if not param in support_dict.keys():
            raise ValueError("the input param is error, \
                only support %s,  but the param is %s" % \
                (str(support_dict.keys()), param))

    def check_support_value(self, param, support_value):
        """check whether the param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        support_value:
            support data
        """
        if param != support_value:
            raise ValueError("the input param is error, \
                only support %s,  but the param is %s" % \
                (support_value, param))

    def check_illegal_value(self, param, illegal_value):
        """check whether the param is correct

        Parameters
        ----------
        param: instance
            the instance to check
        illegal_value:
            don't support data
        """
        if param == illegal_value:
            raise ValueError("the input param is error, \
                don't support %s,  but the param is %s" % \
                (illegal_value, param))

    def encode_list(self, input_list, encode_list):
        """check whether the range of param is correct

        Parameters
        ----------
        input_list: input list
            input param
        encode_list: list
            encoded input param
        """
        encode_list.extend(input_list)
        return encode_list

    def encode(self, params):
        """encode the information of shape to the list of uint32 digit

        Parameters
        ----------
        params: dict of params
            include all information of shape

        Returns
        -------
        params_encode : list of encoded params
            The encoded params, include uint32 numbers
        """
        # encode the dict to list
        if isinstance(params, dict):
            return json.dumps(params)
        else:
            raise TypeError("the type of params only support dict, \
                but the type is %s" % type(params))
