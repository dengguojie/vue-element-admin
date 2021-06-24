#! /usr/bin/env python
# coding=utf-8
"""
Function:
GlobalConfig and WhiteLists class. This class mainly parse the global configuration.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
Change History: 2021-04-12 file Created
"""
import os
from . import utils

WHITE_LIST_FILE_NAME = "white_list_config.json"
FORMAT_ENUM_MAP = "FORMAT_ENUM_MAP"   # FORMAT_ENUM_MAP the map according to graph/types.h
DTYPE_LIST = "DTYPE_LIST"
MINDSPORE_DTYPE_LIST = "MINDSPORE_DTYPE_LIST"
DATA_DISTRIBUTION_LIST = "DATA_DISTRIBUTION_LIST"
AICPU_PROTO2INI_TYPE_MAP = "DTYPE_TO_AICPU_TYPE_MAP"


class WhiteLists:
    """
       the class for white lists, according to st/config/white_list_config.json
    """
    def __init__(self):
        self.format_map = None
        self.type_list = None
        self.mindspore_type_list = None
        self.data_distribution_list = None
        self.aicpu_ir2ini_type_map = None

    def init_white_lists(self):
        """init white lists"""
        config_dir = os.path.join(os.path.dirname(__file__), "..")
        config_path = os.path.join(config_dir, "config", WHITE_LIST_FILE_NAME)
        config_dict = utils.load_json_file(config_path)
        self.format_map = config_dict.get(FORMAT_ENUM_MAP)
        self.type_list = config_dict.get(DTYPE_LIST)
        self.mindspore_type_list = config_dict.get(MINDSPORE_DTYPE_LIST)
        self.data_distribution_list = config_dict.get(DATA_DISTRIBUTION_LIST)
        self.aicpu_ir2ini_type_map = config_dict.get(AICPU_PROTO2INI_TYPE_MAP)
        return config_dict

    def get_aicpu_ir2ini_type_map(self):
        """
        get aicpu_ir2ini_type_map
        """
        return self.aicpu_ir2ini_type_map


class GlobalConfig:
    """
    the class for global config.
    """
    _instance = None

    def __init__(self):
        self.white_lists = None

    @classmethod
    def instance(cls):
        """
        Lazy singleton
        """
        if not cls._instance:
            cls._instance = GlobalConfig.__new__(GlobalConfig)
            white_lists = WhiteLists()
            white_lists.init_white_lists()
            cls._instance.white_lists = white_lists
        return cls._instance

    def get_white_lists(self):
        """get white lists"""
        return self.white_lists
