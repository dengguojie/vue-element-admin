#!/usr/bin/env python
# coding=utf-8
"""
Function:
CaseGenerator class.
This class mainly involves the generate function.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import stat


class ConstManager:
    """
    class ConstManager
    """
    # error code for user:success
    OP_TEST_GEN_NONE_ERROR = 0
    # error code for user: config error
    OP_TEST_GEN_CONFIG_UNSUPPORTED_FMK_TYPE_ERROR = 11
    OP_TEST_GEN_CONFIG_INVALID_OUTPUT_PATH_ERROR = 12
    OP_TEST_GEN_CONFIG_INVALID_OPINFO_FILE_ERROR = 13
    OP_TEST_GEN_CONFIG_INVALID_COMPUTE_UNIT_ERROR = 14
    OP_TEST_GEN_CONFIG_UNSUPPORTED_OPTION_ERROR = 15
    OP_TEST_GEN_CONFIG_OP_DEFINE_ERROR = 16
    OP_TEST_GEN_INVALID_PARAM_ERROR = 17
    OP_TEST_GEN_INVALID_SHEET_PARSE_ERROR = 18
    OP_TEST_GEN_INVALID_DATA_ERROR = 18
    OP_TEST_GEN_AND_RUN_ERROR = 19
    # error code for user: generator error
    OP_TEST_GEN_INVALID_PATH_ERROR = 101
    OP_TEST_GEN_PARSE_JSON_FILE_ERROR = 102
    OP_TEST_GEN_OPEN_FILE_ERROR = 103
    OP_TEST_GEN_CLOSE_FILE_ERROR = 104
    OP_TEST_GEN_OPEN_DIR_ERROR = 105
    OP_TEST_GEN_INDEX_OUT_OF_BOUNDS_ERROR = 106
    OP_TEST_GEN_PARSER_JSON_FILE_ERROR = 107
    OP_TEST_GEN_WRITE_FILE_ERROR = 108
    OP_TEST_GEN_READ_FILE_ERROR = 109
    OP_TEST_GEN_MAKE_DIR_ERROR = 110
    OP_TEST_GEN_GET_KEY_ERROR = 111
    OP_TEST_GEN_MAKE_DIRS_ERROR = 112
    OP_TEST_GEN_INVALID_DEVICE_ID_ERROR = 113
    OP_TEST_GEN_INVALID_INPUT_NAME_ERROR = 114
    OP_TEST_GEN_NONE_TYPICAL_SHAPE_ERROR = 115
    OP_TEST_GEN_INVALID_ERROR_THRESHOLD_ERROR = 116
    # error code for user: un know error
    OP_TEST_GEN_UNKNOWN_ERROR = 1001
    OP_TEST_GEN_TF_LOAD_ERROR = 1002
    OP_TEST_GEN_TF_GET_OPERATORS_ERROR = 1003
    OP_TEST_GEN_TF_WRITE_GRAPH_ERROR = 1004
    OP_TEST_GEN_TF_GET_PLACEHOLDER_ERROR = 1005
    OP_TEST_GEN_TF_CHANGE_PLACEHOLDER_ERROR = 1006

    ATC_TRANSFORM_ERROR = 1110
    ACL_COMPILE_ERROR = 1111
    ACL_TEST_GEN_NONE_ERROR = 0
    ACL_TEST_GEN_ERROR = 255

    BOTH_GEN_AND_RUN_ACL_PROJ = 0
    ONLY_GEN_WITHOUT_RUN_ACL_PROJ = 1
    ONLY_RUN_WITHOUT_GEN_ACL_PROJ = 2
    ONLY_RUN_WITHOUT_GEN_ACL_PROJ_PERFORMANCE = 3
    BOTH_GEN_AND_RUN_ACL_PROJ_PERFORMANCE = 4

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    FMK_LIST = "tf tensorflow caffe"
    SUPPORT_PATH_PATTERN = r"^[A-Za-z0-9_\./:()=\\-]+$"
    EMPTY = ""
    SRC_RELATIVE_TEMPLATE_PATH = "/../template/acl_op_src"

    TESTCASE_CPP_RELATIVE_PATH = "/src/testcase.cpp"
    ACL_OP_JSON_RELATIVE_PATH = "/run/out/test_data/config/acl_op.json"
    TEST_DATA_CONFIG_RELATIVE_PATH = "/run/out/test_data/config"
    TESTCASE_PY_RELATIVE_PATH = "/src/test_{op_name}.py"
    PYTEST_INI_RELATIVE_PATH = "/src/pytest.ini"
    INPUT_SUFFIX_LIST = ['.ini', '.py']
    BIN_FILE = '.bin'
    PY_FILE = '.py'
    FILE_AUTHORITY = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
    FOLDER_MASK = 0o700
    TYPE_UNDEFINED = "UNDEFINED"

    ONLY_GEN_WITHOUT_RUN = 'only_gen_without_run'
    ONLY_RUN_WITHOUT_GEN = 'only_run_without_gen'
    ASCEND_GLOBAL_LOG_LEVEL = 'ascend_global_log_level'
    ASCEND_SLOG_PRINT_TO_STDOUT = 'ascend_slog_print_to_stdout'
    ATC_SINGLEOP_ADVANCE_OPTION = 'atc_singleop_advance_option'
    PERFORMACE_MODE = 'performance_mode'

    # dynamic input.
    DYNAMIC_INPUT = 'dynamic'
    DYNAMIC_INPUT_ARGS = '*dynamic_input'
    DYNAMIC_INPUT_NAME = 'dynamic_input'

    SPACE = ' '
    NEW_LINE_MARK = "\\"
    QUOTATION_MARK = "\""
    COMMA = ','
    IN_OUT_OP_KEY_MAP = {
        'INPUT': 'input',
        'DYNAMIC_INPUT': 'input',
        'OPTIONAL_INPUT': 'input',
        'OUTPUT': 'output',
        'DYNAMIC_OUTPUT': 'output'
    }

    AICPU_ATTR_LIST = ['ATTR', 'REQUIRED_ATTR']
    ATTR_TYPE_MAP = {
        'int': 'int',
        'float': 'float',
        'bool': 'bool',
        'str': 'string',
        'type': 'data_type',
        'listInt': 'list_int',
        'listFloat': 'list_float',
        'listBool': 'list_bool',
        'listStr': 'list_string',
        'listListInt': 'list_list_int'
    }

    OP_ATTR_TYPE_MAP = {
        'int': 'OP_INT',
        'float': 'OP_FLOAT',
        'bool': 'OP_BOOL',
        'string': 'OP_STRING',
        'data_type': 'OP_DTYPE',
        'list_int': 'OP_LIST_INT',
        'list_float': 'OP_LIST_FLOAT',
        'list_bool': 'OP_LIST_BOOL',
        'list_string': 'OP_LIST_STRING',
        'list_list_int': 'OP_LIST_INT_PTR'
    }

    OP_PROTO_PARSE_ATTR_TYPE_MAP = {
        "Int": "int",
        "Float": "float",
        "String": "str",
        "Bool": "bool",
        "Type": "type",
        "ListInt": "listInt",
        "ListFloat": "listFloat",
        "ListString": "listStr",
        "ListBool": "listBool",
        "ListListInt": "listListInt"
    }

    ATTR_MEMBER_VAR_MAP = {
        'int': 'intAttr',
        'float': 'floatAttr',
        'bool': 'boolAttr',
        'string': 'stringAttr',
        'data_type': 'dtypeAttr',
        'list_int': 'listIntAttr',
        'list_float': 'listFloatAttr',
        'list_bool': 'listBoolAttr',
        'list_string': 'listStringAttr',
        'list_list_int': 'listIntPtrAttr'
    }

    ATTR_TYPE_SUPPORT_TYPE_MAP = {
        "int8": "DT_INT8",
        "int32": "DT_INT32",
        "int16": "DT_INT16",
        "int64": "DT_INT64",
        "uint8": "DT_UINT8",
        "uint16": "DT_UINT16",
        "uint32": "DT_UINT32",
        "uint64": "DT_UINT64",
        "float": "DT_FLOAT",
        "float16": "DT_FLOAT16",
        "float32": "DT_FLOAT",
        "bool": "DT_BOOL",
        "double": "DT_DOUBLE",
        "complex64": "DT_COMPLEX64",
        "complex128": "DT_COMPLEX128"
    }

    FALSE = 'false'

    # ---------------------------GlobalConfig--------------------------
    WHITE_LIST_FILE_NAME = "white_list_config.json"
    FORMAT_ENUM_MAP = "FORMAT_ENUM_MAP"  # FORMAT_ENUM_MAP the map according to graph/types.h
    DTYPE_LIST = "DTYPE_LIST"
    MINDSPORE_DTYPE_LIST = "MINDSPORE_DTYPE_LIST"
    DATA_DISTRIBUTION_LIST = "DATA_DISTRIBUTION_LIST"
    AICPU_PROTO2INI_TYPE_MAP = "DTYPE_TO_AICPU_TYPE_MAP"

    # ----------------------------model_parser---------------------------
    GET_MODEL_NODES_FUNC = 'get_model_nodes'
    GET_SHAPE_FUNC = 'get_shape'
    CHANGE_SHAPE_FUNC = 'change_shape'
    FILE_NAME_SUFFIX = '_model_parser'
    FRAMEWORK_CONFIG_PATH = './framework/framework.json'

    # ----------------------------CaseDesign--------------------------
    OP = 'op'
    INPUT_DESC = 'input_desc'
    OUTPUT_DESC = 'output_desc'
    VALUE_RANGE = 'value_range'
    DATA_DISTRIBUTE = 'data_distribute'
    ATTR = 'attr'
    CASE_NAME = 'case_name'
    ST_MODE = 'st_mode'
    FUZZ_IMPL = 'fuzz_impl'
    REQUIRED_KEYS = [OP, INPUT_DESC, OUTPUT_DESC, CASE_NAME]
    # --------------------------CaseGenerator---------------
    INI_INPUT = 'input'
    INI_OUTPUT = 'output'
    IO_TYPE = ['inputs', 'outputs']
    OP_NAME = 'op_name'
    REQUIRED_OP_INFO_KEYS = ["paramType", "name"]
    PARAM_TYPE_VALID_VALUE = ["dynamic", "optional", "required"]
    OPTIONAL_TYPE_LIST = ['UNDEFINED', 'RESERVED']
    TRUE_OR_FALSE_LIST = ['True', 'False']
    # dynamic shape scenario add keys as follows
    SHAPE_RANGE = 'shape_range'
    TYPICAL_SHAPE = 'typical_shape'
    # Two dynamic scenarios: shape value is -1 or -2
    SHAPE_DYNAMIC_SCENARIOS_ONE = -1
    SHAPE_DYNAMIC_SCENARIOS_TWO = -2
    # dynamic shape scenario, shape_range default value.
    SHAPE_RANGE_DEFAULT_VALUE = [[1, -1]]
    VALUE = 'value'
    IS_CONST = 'is_const'
    CONST_VALUE = 'const_value'

    # --------------------------SubCaseDesign-----------------------
    ATTR_REQUIRED_KEYS = ["name", "type", "value"]
    # ---------------------------SubCaseDesignCross-----------------
    # due to orthogonal combination, type need to behind shape.
    INPUT_CROSS_LIST = ['format', 'shape', 'type', 'data_distribute',
                        'value_range']
    # due to orthogonal combination, type need to behind shape, also.
    OUTPUT_CROSS_LIST = ['format', 'shape', 'type']
    MS_INPUT_CROSS_LIST = ['type', 'shape', 'data_distribute', 'value_range']
    MS_OUTPUT_CROSS_LIST = ['type', 'shape']

    # ---------------------------SubCaseDesignFuzz-------------------
    FUZZ_CASE_NUM = 'fuzz_case_num'
    FUZZ_FUNCTION = 'fuzz_branch'
    MAX_FUZZ_CASE_NUM = 2000

    # ----------------------------AclOpRunner--------------------------
    CMAKE_LIST_FILE_NAME = 'CMakeLists.txt'
    BUILD_INTERMEDIATES_HOST = 'build/intermediates/host'
    RUN_OUT = 'run/out'
    MAIN = 'main'
    PROF = 'prof'
    INSTALL_PATH = 'install_path'
    MSPROF_REL_PATH = '/toolkit/tools/profiler/bin/msprof'
    MSPROF_PYC_REL_PATH = '/toolkit/tools/profiler/profiler_tool/analysis/msprof/msprof.py'
    SUMMARY_REL_PATH = 'summary'
    OP_SUMMARY_CSV = 'op_summary_0_1.csv'
    PROF_PYTHON_CMD = "python3.7"
    NULL_RESULT_FILE_LINE_NUM = 2
    RESULT_FILE_COLUMN_NUM = 3
    RESULT_FILE_CASE_NAME_COLUMN_NUM = 1
    SHOW_DATA_UPPER_LIMLT = 20
    SHOW_TOP_TEN_DATA = 10
    SHOW_LAST_TEN_DATA = 10
    PROF_TIME_UNIT = 'us'
    OP_CASE_INFO_IN_CSV_COLUMN_NAME_LIST = [
        'Op Name', 'Task Type', 'Task Duration(us)', 'Task ID']
    OP_NAME_INDEX = 0
    TASK_TYPE_INDEX = 1
    TASK_DURATION_INDEX = 2
    TASK_ID_INDEX = 3
    OP = 'op'

    # -----------------MsOpRunner------------------------
    TEST_PY = 'test_{op_name}.py'
    # ------------------AdvanceIniArgs---------------------
    ADVANCE_SECTION = 'RUN'
    ASCEND_GLOBAL_LOG_LEVEL_LIST = ['0', '1', '2', '3', '4']
    ASCEND_SLOG_PRINT_TO_STDOUT_LIST = ['0', '1']

    # --------------------st_report-----------------------
    DATA_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    DATA_FILE_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP
    DATA_DIR_MODES = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
    EXPECT_SUCCESS = "success"
    EXPECT_FAILED = "failed"

    def get_op_name(self):
        """
        get operator name
        :return: op_name
        """
        return self.OP_NAME

    def get_case_name(self):
        """
        get case name
        :return: case_name
        """
        return self.CASE_NAME
