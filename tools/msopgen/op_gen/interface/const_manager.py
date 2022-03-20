#!/usr/bin/env python
# coding=utf-8
"""
Function:
This file mainly involves the common function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""


import os
import stat


class ConstManager:
    """
    The class for const manager
    """
    # error code for user:success
    MS_OP_GEN_NONE_ERROR = 0
    # error code for user: config error
    MS_OP_GEN_CONFIG_UNSUPPORTED_FMK_TYPE_ERROR = 11
    MS_OP_GEN_CONFIG_INVALID_OUTPUT_PATH_ERROR = 12
    MS_OP_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR = 13
    MS_OP_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR = 14
    MS_OP_GEN_CONFIG_UNSUPPORTED_MODE_ERROR = 15
    MS_OP_GEN_CONFIG_OP_DEFINE_ERROR = 16
    MS_OP_GEN_INVALID_PARAM_ERROR = 17
    MS_OP_GEN_INVALID_SHEET_PARSE_ERROR = 18
    # error code for user: generator error
    MS_OP_GEN_INVALID_PATH_ERROR = 101
    MS_OP_GEN_PARSE_DUMP_FILE_ERROR = 102
    MS_OP_GEN_OPEN_FILE_ERROR = 103
    MS_OP_GEN_CLOSE_FILE_ERROR = 104
    MS_OP_GEN_OPEN_DIR_ERROR = 105
    MS_OP_GEN_INDEX_OUT_OF_BOUNDS_ERROR = 106
    MS_OP_GEN_PARSER_JSON_FILE_ERROR = 107
    MS_OP_GEN_WRITE_FILE_ERROR = 108
    MS_OP_GEN_READ_FILE_ERROR = 109
    MS_OP_GEN_UNKNOWN_CORE_TYPE_ERROR = 110
    MS_OP_GEN_PARSER_EXCEL_FILE_ERROR = 111
    MS_OP_GEN_JSON_DATA_ERROR = 112
    MS_OP_GEN_INVALID_FILE_ERROR = 113
    # error code for user: un know error
    MS_OP_GEN_UNKNOWN_ERROR = 1001
    # call os/sys error:
    MS_OP_GEN_MAKE_DIRS_ERROR = 1002
    MS_OP_GEN_COPY_DIRS_ERROR = 1003
    MS_OP_GEN_IMPORT_MODULE_ERROR = 1004

    LEFT_BRACES = "{"
    RIGHT_BRACES = "}"
    SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"
    FMK_MS = ["ms", "mindspore"]
    FMK_LIST = ["tf", "tensorflow", "caffe", "pytorch", "ms", "mindspore", "onnx"]
    PROJ_MS_NAME = "mindspore"
    GEN_MODE_LIST = ['0', '1']
    OP_TEMPLATE_PATH = "../template/op_project_tmpl"
    MS_PROTO_PATH = "op_proto"
    OP_TEMPLATE_MS_OP_PROTO_PATH = "../template/op_project_tmpl/op_proto"
    OP_TEMPLATE_AICPU_PATH = "../template/cpukernel"
    OP_TEMPLATE_TBE_PATH = "../template/tbe"
    SPACE = ' '
    EMPTY = ''
    FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
    FOLDER_MASK = 0o700

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    # path
    IMPL_DIR = "tbe/impl/"
    MS_IMPL_DIR = "mindspore/impl"
    IMPL_NAME = "_impl"
    IMPL_SUFFIX = ".py"

    # input arguments
    INPUT_ARGUMENT_CMD_GEN = 'gen'
    INPUT_ARGUMENT_CMD_MI = 'mi'
    INPUT_ARGUMENT_CMD_MI_QUERY = 'query'

    OP_INFO_WITH_PARAM_TYPE_LEN = 3
    OP_INFO_WITH_FORMAT_LEN = 4
    AICPU_CORE_TYPE_LIST = ['aicpu', 'ai_cpu']
    AICORE_CORE_TYPE_LIST = ['aicore', 'ai_core', 'vectorcore', 'vector_core']
    PARAM_TYPE_DYNAMIC = "dynamic"
    PARAM_TYPE_REQUIRED = "required"
    PARAM_TYPE_OPTIONAL = "optional"
    PARAM_TYPE_MAP_INI = {"1": PARAM_TYPE_REQUIRED, "0": PARAM_TYPE_OPTIONAL}
    INPUT_OUTPUT_PARAM_TYPE = [PARAM_TYPE_DYNAMIC, PARAM_TYPE_REQUIRED,
                               PARAM_TYPE_OPTIONAL]
    ATTR_PARAM_TYPE = [PARAM_TYPE_REQUIRED, PARAM_TYPE_OPTIONAL]

    # input file type
    INPUT_FILE_XLSX = ".xlsx"
    INPUT_FILE_XLS = ".xls"
    INPUT_FILE_TXT = ".txt"
    INPUT_FILE_JSON = ".json"
    INPUT_FILE_EXCEL = (INPUT_FILE_XLSX, INPUT_FILE_XLS)
    MI_VALID_TYPE = (INPUT_FILE_XLSX, INPUT_FILE_XLS, INPUT_FILE_JSON)
    GEN_VALID_TYPE = (INPUT_FILE_XLSX, INPUT_FILE_XLS, INPUT_FILE_TXT,
                      INPUT_FILE_JSON)

    # keys in map
    INFO_IR_TYPES_KEY = "ir_type_list"
    INFO_PARAM_TYPE_KEY = "param_type"
    INFO_PARAM_FORMAT_KEY = "format_list"

    # GenModeType
    GEN_PROJECT = '0'
    GEN_OPERATOR = '1'

    # CoreType
    AICORE = 0
    AICPU = 1

    def get_aicore(self: any) -> int:
        """
        get ai_core flag
        :return: ai_core flag
        """
        return self.AICORE

    def get_aicpu(self: any) -> int:
        """
        get aicpu flag
        :return: aicpu flag
        """
        return self.AICPU
