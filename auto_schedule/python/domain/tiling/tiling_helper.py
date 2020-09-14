"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

Tiling helper, set and get a fixed tiling
"""
import os
import copy
import json
from te import tvm


class Singleton():
    """Singleton base class
    """
    __instance = None
    __tiling = None
    __tiling_type = None
    __input_params = None

    def __init__(self):
        self._singleton__tiling = dict()
        self._singleton__tiling_type = None
        self._singleton__input_params = None

        get_config_path_fun = tvm.get_global_func("_query_config_path")
        config_path = get_config_path_fun()
        config_path = os.path.realpath(config_path)
        if os.path.exists(config_path):
            with open(config_path, 'r') as handler:
                config = json.load(handler)
                self._singleton__tiling_type = config.get("tiling_type")
        if not self._singleton__tiling_type:
            self._singleton__tiling_type = "auto_tiling"

    def __new__(cls, *args, **kw):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kw)
        return cls.__instance

    def get_params(self):
        """Get tiling input params

        Notice
        ------
        this function is create for auto tune tool to get tiling input params

        Returns
        ----------
        input_params: dict
            set by tiling query or get tiling
        """
        return copy.deepcopy(self._singleton__input_params)

    def set_params(self, inputs):
        """Set get tiling or tiling query input params

        Parameters
        ----------
        inputs: dict
            build up by schedule

        Notice
        ------
        this function is create for auto tune tool to get tiling input params,
            params set last time should be same to get_tiling inputs, usually
            called under non-tuning_tiling mode by schedule when building
            executable binary file

        """
        if isinstance(inputs, dict) and \
            isinstance(inputs.get("kernel_name", None), tvm.expr.StringImm):
            inputs["kernel_name"] = inputs['kernel_name'].value
        self._singleton__input_params = copy.deepcopy(inputs)

    def get_tiling(self, inputs):
        """Get the tiling from Singleton object
        Parameters
        ----------

        Notice
        ------
        this function is work under tuning tiling mode together with
            set tiling, input params used is set by set_params last
            time, should be exaclty same to inputs

        some list value given is tvm.expr.Int, so compare use string
            and not original dict

        Returns
        -------
        tiling: dict
            The tiling saved in Singleton object
        """
        _kernel_name = inputs['kernel_name']
        if isinstance(_kernel_name, tvm.expr.StringImm):
            _kernel_name = inputs['kernel_name'].value
        if not isinstance(inputs, dict) or not inputs:
            raise RuntimeError("illegal inputs: %s" % str(inputs))

        if not isinstance(inputs, dict) or not self._singleton__input_params:
            raise RuntimeError("set params when tuning tiling, like: %s"
                               % str(inputs))

        ignore_list = ["reserved_ub", "kernel_name", "op_name", "test_case"]
        pre_params = dict()
        cur_params = dict()
        for key in self._singleton__input_params:
            if key not in ignore_list:
                pre_params[key] = self._singleton__input_params[key]

        for key in inputs:
            if key not in ignore_list:
                cur_params[key] = inputs[key]

        if str(cur_params) != str(pre_params):
            raise RuntimeError("tiling params is changed, previous is: %s, "
                               "input is %s"
                               % (str(pre_params), str(cur_params)))

        return copy.deepcopy(self._singleton__tiling.get(_kernel_name, None))

    def set_tiling(self, tiling, kernel_name="kernel_name"):
        """Set the tiling to private member variable of Singleton object
        Parameters
        ----------
        tiling: dict
            The setting tiling

        Returns
        -------
        """
        type_list = ["tuning_tiling"]
        if self._singleton__tiling_type not in type_list:
            raise RuntimeError("tiling mode is not tuning tiling, "
                               "current is %s"
                               % str(self._singleton__tiling_type))

        if isinstance(tiling, dict):
            self._singleton__tiling[kernel_name] = copy.deepcopy(tiling)
        else:
            raise TypeError('tiling is not a dict.')

    def get_tiling_type(self):
        """Get the tiling type from Singleton object
        Parameters
        ----------

        Returns
        -------
        tiling_type: string
            The tiling type saved in Singleton object
        """
        return copy.deepcopy(self._singleton__tiling_type)

    def set_tiling_type(self, tiling_type):
        """Set the tiling type to private member variable of Singleton object
        Parameters
        ----------
        tiling_type: string
            The setting tiling type

        Returns
        -------
        """
        if isinstance(tiling_type, str):
            self._singleton__tiling_type = copy.deepcopy(tiling_type)
        else:
            raise TypeError('tiling is not a str.')

    def instance_refresh(self, tiling_type=None, \
                         input_params=None, tiling_dict=None):
        """refresh private member variable of Singleton object
        Parameters
        ----------
        tiling_type: string
            The setting tiling type
        input_params:dict
        tiling_dict: tiling dict
        Returns
        -------
        """
        self._singleton__tiling = copy.deepcopy(tiling_dict)
        self._singleton__tiling_type = copy.deepcopy(tiling_type)
        self._singleton__input_params = copy.deepcopy(input_params)

TILING_INSTANCE = Singleton()
