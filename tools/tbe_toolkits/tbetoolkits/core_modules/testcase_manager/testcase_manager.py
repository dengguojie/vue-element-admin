#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Universal testcase manager for csv support
"""
# Standard Packages
import csv
import hashlib
import random
import logging
from enum import auto
from enum import Enum
from typing import TextIO
from typing import Callable
from typing import Optional
from typing import Dict
from typing import Tuple
from typing import Any
from typing import NoReturn

# Third-Party Packages
import tbetoolkits
from .field_parser import process_string
from .field_parser import shapelike_stc
from .field_parser import shapelike_stc_ex
from .field_parser import shapelike_float
from .field_parser import shapelike_float_signed
from .field_parser import string_container
from .field_parser import int_container
from .field_parser import process_int
from .field_parser import process_float
from .field_parser import process_bool
from .field_parser import process_dict
from .field_parser import process_dynamic_shapelike
from .field_parser import process_dynamic_inferable_shapelike
from .field_parser import process_eval
from .field_parser import rangelike
from ...utilities import get
from ...utilities import get_dtype_width
from ...utilities import shape_product
from ...utilities import parse_dtype
from ...utilities import get_networkname_by_testcase_name
from ...utilities import set_process_name
from ...utilities import set_thread_name
from ...utilities import get_global_storage
from ...utilities import eliminate_scalar_shapes
from ..infershape.infershape import shape_inference
from ..infershape.infershape import tensorflow_inference


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


class PLACEHOLDER:
    """
    Simple Placeholder
    """
    pass


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


class UniversalTestcaseStructure:
    """
    Structure for UniversalTestcase Profiling
    """

    __slots__ = ("dyn_kernel_name",
                 "stc_kernel_name",
                 "cst_kernel_name",
                 "bin_kernel_name",
                 "dyn_inputs",
                 "dyn_outputs",
                 "dyn_input_formats",
                 "dyn_block_dim",
                 "dyn_workspaces",
                 "dyn_obj_size",
                 "dyn_kernel_size",
                 "dyn_tiling_key",
                 "dyn_sch_count",
                 "dyn_func_params",
                 "dyn_compile_info",
                 "dyn_compile_result",
                 "dyn_compile_time",
                 "dyn_op_pattern",
                 "dyn_tiling_data",
                 "dyn_tiling_time",
                 "bin_block_dim",
                 "bin_workspaces",
                 "bin_obj_size",
                 "bin_kernel_size",
                 "bin_tiling_key",
                 "bin_sch_count",
                 "bin_compile_info",
                 "bin_compile_result",
                 "bin_compile_time",
                 "bin_tiling_data",
                 "bin_tiling_time",
                 "stc_block_dim",
                 "stc_workspaces",
                 "stc_compile_result",
                 "stc_compile_time",
                 "stc_rl_query_result",
                 "stc_op_pattern",
                 "stc_input_as_list_distribution",
                 "stc_op_name",
                 "cst_block_dim",
                 "cst_workspaces",
                 "cst_compile_result",
                 "cst_compile_time",
                 "cst_op_pattern",
                 "cst_rl_status",
                 "output_as_list_distribution",
                 "op_name",
                 "shape_check",
                 "stc_inputs",
                 "dyn_input_dtypes",
                 "stc_input_dtypes",
                 "stc_input_formats",
                 "input_data_ranges",
                 "other_runtime_params",
                 "stc_ori_inputs",
                 "stc_input_ori_formats",
                 "stc_outputs",
                 "output_dtypes",
                 "output_formats",
                 "stc_ori_outputs",
                 "output_ori_formats",
                 "precision_tolerances",
                 "random_buff",
                 "input_as_variable",
                 "dump_input_data_name",
                 "dump_output_data_name",
                 "manual_input_data_binaries",
                 "manual_output_data_binaries",
                 "manual_tiling_data",
                 "manual_dyn_workspaces",
                 "manual_stc_workspaces",
                 "manual_block_dim",
                 "manual_tiling_op_type",
                 "manual_stc_build_config",
                 "input_arrays",
                 "original_input_arrays",
                 "stc_input_byte_arrays",
                 "dyn_input_arrays",
                 "dyn_input_byte_arrays",
                 "actual_input_data_ranges",
                 "golden_arrays",
                 "output_byte_arrays",
                 "dyn_workspace_byte_arrays",
                 "stc_workspace_byte_arrays",
                 "cst_workspace_byte_arrays",
                 "bin_workspace_byte_arrays",
                 "model",
                 "device_id",
                 "bin_is_tik",
                 "is_tik",
                 "dyn_tuple_tiling_data",
                 "dyn_str_tiling_data",
                 "dyn_tiling_data_bytes",
                 "bin_tuple_tiling_data",
                 "bin_str_tiling_data",
                 "bin_tiling_data_bytes",
                 "bin_dyn_tiling_key",
                 "dyn_prof_result",
                 "stc_prof_result",
                 "cst_prof_result",
                 "bin_prof_result",
                 "dyn_input_ori_formats",
                 "dyn_input_ranges",
                 "dyn_output_ranges",
                 "dyn_ori_inputs",
                 "dyn_ori_outputs",
                 "dyn_input_as_list_distribution",
                 "manual_dyn_build_config",
                 "other_compilation_params",
                 "const_input_indexes",
                 "const_input_modes",
                 "output_inplace_indexes",
                 "testcase_name",
                 "network_name",
                 "strict_precision_mode",
                 "absolute_precision",
                 "is_enabled",
                 "original_line",
                 "original_header",
                 "dyn_fail_reason",
                 "fail_reason",
                 "dyn_is_valid",
                 "is_valid",
                 "ready_for_profile",
                 "torch_func")
    identity_headers: Dict[str, tuple] = {
        "testcase_name": (FIELD_TYPES.STRING, None),  # Required
        "network_name": (FIELD_TYPES.STRING, None, None),
    }
    dynamic_property_headers: Dict[str, tuple] = {
        "dyn_inputs": (FIELD_TYPES.SHAPELIKE_DYN, None, None),
        "dyn_input_dtypes": (FIELD_TYPES.STRING_CONTAINER, None, None),
        "dyn_outputs": (FIELD_TYPES.SHAPELIKE_DYN_EX, None, None),
        "dyn_ori_inputs": (FIELD_TYPES.SHAPELIKE_DYN, ("dyn_inputs",), None),
        "dyn_ori_outputs": (FIELD_TYPES.SHAPELIKE_DYN_EX, ("dyn_outputs",), None),
        "dyn_input_formats": (FIELD_TYPES.STRING_CONTAINER, None, ("ND",)),
        "dyn_input_ori_formats": (FIELD_TYPES.STRING_CONTAINER, ("stc_input_ori_formats",), ("ND",)),
        "dyn_input_ranges": (FIELD_TYPES.RANGELIKE, None, ()),
        "dyn_output_ranges": (FIELD_TYPES.RANGELIKE, None, ()),
    }
    non_platform_static_property_headers: Dict[str, tuple] = {
        "op_name": (FIELD_TYPES.STRING, None),  # Required
        "stc_input_dtypes": (FIELD_TYPES.STRING_CONTAINER, None),  # Required
        "stc_ori_inputs": (FIELD_TYPES.SHAPELIKE_STC, ("stc_inputs",)),
        "stc_ori_outputs": (FIELD_TYPES.SHAPELIKE_STC_EX, ("stc_outputs",), None),
        "stc_input_ori_formats": (FIELD_TYPES.STRING_CONTAINER, ("dyn_input_ori_formats",), ("ND",)),
        "output_ori_formats": (FIELD_TYPES.STRING_CONTAINER, ("output_formats",), ("ND",)),
        "other_compilation_params": (FIELD_TYPES.DICT, None, {}),
        "other_runtime_params": (FIELD_TYPES.DICT, None, {}),
    }
    static_property_headers: Dict[str, tuple] = {
        "stc_inputs": (FIELD_TYPES.SHAPELIKE_STC, None),  # Required
        "output_dtypes": (FIELD_TYPES.STRING_CONTAINER, None),  # Required
        "stc_outputs": (FIELD_TYPES.SHAPELIKE_DYN_EX, None, None),
        "stc_input_formats": (FIELD_TYPES.STRING_CONTAINER, ("dyn_input_formats",), ("ND",)),
        "output_formats": (FIELD_TYPES.STRING_CONTAINER, ("output_ori_formats",), ("ND",)),
    }
    special_property_headers: Dict[str, tuple] = {
        "dyn_input_as_list_distribution": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "stc_input_as_list_distribution": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "output_as_list_distribution": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "input_as_variable": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "stc_op_name": (FIELD_TYPES.STRING, ("op_name",)),
        "const_input_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "const_input_modes": (FIELD_TYPES.STRING_CONTAINER, None, (None,)),
        "precision_tolerances": (FIELD_TYPES.SHAPELIKE_FLOAT, None, ((0.001, 0.001),)),
        "input_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED, None, (None, None)),
        "strict_precision_mode": (FIELD_TYPES.BOOL, None, True),
        "absolute_precision": (FIELD_TYPES.FLOAT, None, 0),
        "shape_check": (FIELD_TYPES.BOOL, None, True),
        "output_inplace_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "random_buff": (FIELD_TYPES.BOOL, None, False)
    }
    property_headers: Dict[str, tuple] = {
        **dynamic_property_headers, **non_platform_static_property_headers, **static_property_headers,
        **special_property_headers
    }
    option_headers: Dict[str, tuple] = {
        # Manually controlled property
        "is_enabled": (FIELD_TYPES.BOOL, None, True),
        "dump_input_data_name": (FIELD_TYPES.STRING, None, None),
        "dump_output_data_name": (FIELD_TYPES.STRING, None, None),
        "manual_input_data_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
        "manual_output_data_binaries": (FIELD_TYPES.STRING_CONTAINER, None, ()),
        "manual_tiling_data": (FIELD_TYPES.INT_CONTAINER, None, None),
        "manual_dyn_workspaces": (FIELD_TYPES.INT_CONTAINER, None, None),
        "manual_stc_workspaces": (FIELD_TYPES.INT_CONTAINER, None, None),
        "manual_block_dim": (FIELD_TYPES.INT_CONTAINER, None, None),
        "manual_tiling_op_type": (FIELD_TYPES.STRING, None, None),
        "manual_dyn_build_config": (FIELD_TYPES.DICT, None, {"save_temp_cce_file": True}),
        "manual_stc_build_config": (FIELD_TYPES.DICT, None, {"save_temp_cce_file": True})}
    complete_headers: Dict[str, tuple] = {**identity_headers, **property_headers, **option_headers}
    type_processing_func: Dict[FIELD_TYPES, Callable] = {
        FIELD_TYPES.STRING: process_string,
        FIELD_TYPES.SHAPELIKE_DYN: process_dynamic_shapelike,
        FIELD_TYPES.SHAPELIKE_DYN_EX: process_dynamic_inferable_shapelike,
        FIELD_TYPES.RANGELIKE: rangelike,
        FIELD_TYPES.SHAPELIKE_STC: shapelike_stc,
        FIELD_TYPES.SHAPELIKE_STC_EX: shapelike_stc_ex,
        FIELD_TYPES.SHAPELIKE_FLOAT: shapelike_float,
        FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED: shapelike_float_signed,
        FIELD_TYPES.STRING_CONTAINER: string_container,
        FIELD_TYPES.INT_CONTAINER: int_container,
        FIELD_TYPES.INT: process_int,
        FIELD_TYPES.FLOAT: process_float,
        FIELD_TYPES.BOOL: process_bool,
        FIELD_TYPES.DICT: process_dict,
        FIELD_TYPES.FREE_EVAL: process_eval}

    def __init__(self):
        super().__init__()
        self.dyn_kernel_name: Optional[str] = None
        self.stc_kernel_name: Optional[str] = None
        self.cst_kernel_name: Optional[str] = None
        self.bin_kernel_name: Optional[str] = None
        # DYN
        self.dyn_inputs: Optional[tuple] = None
        self.dyn_outputs: Optional[tuple] = None
        self.dyn_input_formats: Optional[tuple] = None
        self.dyn_input_ori_formats: Optional[tuple] = None
        self.dyn_input_ranges: Optional[tuple] = None
        self.dyn_output_ranges: Optional[tuple] = None
        self.dyn_ori_inputs: Optional[tuple] = None
        self.dyn_ori_outputs: Optional[tuple] = None
        self.dyn_input_as_list_distribution: Optional[tuple] = None
        # DYN_RUNTIME
        self.dyn_block_dim: Optional[str] = None
        self.dyn_workspaces: Optional[tuple] = None
        self.dyn_obj_size = None
        self.dyn_kernel_size = None
        self.dyn_tiling_key: Optional[int] = None
        self.dyn_sch_count = None
        self.dyn_func_params = None
        self.dyn_compile_info = None
        self.dyn_compile_result = None
        self.dyn_compile_time = None
        self.dyn_tiling_data: Optional[bytes] = None
        self.dyn_tiling_time: Optional[str] = None
        self.dyn_op_pattern: Optional[str] = None
        # BIN
        self.bin_block_dim: Optional[str] = None
        self.bin_workspaces: Optional[tuple] = None
        self.bin_obj_size = None
        self.bin_kernel_size = None
        self.bin_tiling_key: Optional[int] = None
        self.bin_sch_count = None
        self.bin_compile_info = None
        self.bin_compile_result = None
        self.bin_compile_time = None
        self.bin_tiling_data: Optional[bytes] = None
        self.bin_tiling_time: Optional[str] = None
        # STC
        self.stc_block_dim: Optional[str] = None
        self.stc_workspaces: Optional[tuple] = None
        self.stc_compile_result = None
        self.stc_compile_time = None
        self.stc_rl_query_result = None
        self.stc_op_pattern = None
        self.stc_input_as_list_distribution = None
        self.stc_op_name = None
        # CST
        self.cst_block_dim: Optional[str] = None
        self.cst_workspaces: Optional[tuple] = None
        self.cst_compile_result = None
        self.cst_compile_time = None
        self.cst_op_pattern = None
        self.cst_rl_status = None
        # Others
        self.output_as_list_distribution = None
        self.op_name: Optional[str] = None
        self.shape_check: Optional[bool] = None
        self.stc_inputs: Optional[tuple] = None
        self.dyn_input_dtypes: Optional[tuple] = None
        self.stc_input_dtypes: Optional[tuple] = None
        self.stc_input_formats = None
        self.input_data_ranges = None
        self.other_runtime_params: Optional[dict] = None
        self.stc_ori_inputs = None
        self.stc_input_ori_formats = None
        self.stc_outputs = None
        self.output_dtypes = None
        self.output_formats = None
        self.stc_ori_outputs = None
        self.output_ori_formats = None
        self.precision_tolerances = None
        self.random_buff = None
        self.input_as_variable = None
        # Manual controlled parameters
        self.dump_input_data_name = None
        self.dump_output_data_name = None
        self.manual_input_data_binaries: Optional[Tuple[str, ...]] = None
        self.manual_output_data_binaries = None
        self.manual_tiling_data = None
        self.manual_dyn_workspaces = None
        self.manual_stc_workspaces = None
        self.manual_block_dim = None
        self.manual_tiling_op_type = None
        self.manual_stc_build_config = None

        self.input_arrays = None
        self.original_input_arrays = None
        self.stc_input_byte_arrays = None
        self.dyn_input_arrays = None
        self.dyn_input_byte_arrays = None
        self.actual_input_data_ranges = None
        self.golden_arrays = None
        self.output_byte_arrays = None
        self.dyn_workspace_byte_arrays = None
        self.stc_workspace_byte_arrays = None
        self.cst_workspace_byte_arrays = None
        self.bin_workspace_byte_arrays = None
        # Switch
        self.model = None
        self.device_id = None
        # Tiling
        self.bin_is_tik = None
        self.is_tik = None
        self.dyn_tuple_tiling_data = None
        self.dyn_str_tiling_data = None
        self.dyn_tiling_data_bytes = None
        self.bin_tuple_tiling_data = None
        self.bin_str_tiling_data = None
        self.bin_tiling_data_bytes = None
        self.bin_dyn_tiling_key = None
        # Outputs
        self.dyn_prof_result: Optional["tbetoolkits.core_modules.profiling.profiling.RTSProfilingResult"] = None
        self.stc_prof_result: Optional["tbetoolkits.core_modules.profiling.profiling.RTSProfilingResult"] = None
        self.cst_prof_result: Optional["tbetoolkits.core_modules.profiling.profiling.RTSProfilingResult"] = None
        self.bin_prof_result: Optional["tbetoolkits.core_modules.profiling.profiling.RTSProfilingResult"] = None
        # Temp
        self.manual_dyn_build_config: Optional[dict] = None
        self.other_compilation_params: Optional[dict] = None
        self.const_input_indexes: Optional[tuple] = None
        self.const_input_modes = None
        self.output_inplace_indexes = None
        self.testcase_name: Optional[str] = None
        self.network_name: Optional[str] = None
        self.strict_precision_mode = None
        self.absolute_precision = None
        self.is_enabled: bool = True
        self.original_line: Optional[tuple] = None
        self.original_header: Optional[tuple] = None

        # Test Runtime Attributes
        self.dyn_fail_reason: Optional[str] = None
        self.fail_reason: Optional[str] = None
        self.dyn_is_valid: Optional[bool] = True
        self.is_valid: Optional[bool] = True
        self.ready_for_profile: Optional[int] = 0
        self.torch_func: Optional[Callable] = None

    def get_compilation_hash(self, is_binary: Optional[bool] = False) -> int:
        """
        Get compilation related param hash
        :return: Optional[int] Hash
        """
        compilation_params = (
            self.op_name,
            self.dyn_inputs if not is_binary else "BINARY",
            self.dyn_outputs if not is_binary else "BINARY",
            self.dyn_input_dtypes,
            self.output_dtypes,
            self.dyn_input_formats,
            self.dyn_input_ori_formats,
            self.output_formats,
            self.output_ori_formats,
            self.dyn_input_ranges if not is_binary else "BINARY",
            self.dyn_output_ranges if not is_binary else "BINARY",
            self.dyn_ori_inputs if not is_binary else "BINARY",
            self.dyn_ori_outputs if not is_binary else "BINARY",
            self.dyn_input_as_list_distribution,
            self.output_as_list_distribution,
            self.const_input_indexes,
            self.const_input_modes,
            self.manual_dyn_build_config,
            self.other_compilation_params,
            self.random_buff)
        return hash(str(compilation_params))

    def as_dict(self) -> dict:
        titles = tuple(self.identity_headers.keys() + tuple(self.property_headers.keys()))
        values = tuple(getattr(self, title) for title in titles)
        return dict(zip(titles, values))

    def get_hash(self) -> bytes:
        """
        GPU Testcase uniqueness
        """
        values = tuple(getattr(self, title) for title in tuple(self.non_platform_static_property_headers.keys()))
        result_dict = dict(zip(tuple(self.non_platform_static_property_headers.keys()), values))
        return hashlib.blake2b(str(result_dict).encode('UTF-8')).digest()

    @staticmethod
    def from_dict(dict_testcase: dict):
        result = UniversalTestcaseStructure()
        for key in dict_testcase:
            setattr(result, key, dict_testcase[key])
        return result

    def __hash__(self):
        return hash(self.testcase_name)

    def ready(self) -> bool:
        """Ready for profile"""
        if self.ready_for_profile == 4:
            return True
        return False

    @staticmethod
    def from_dict(dict_testcase: dict):
        result = UniversalTestcaseStructure()
        for key in dict_testcase:
            if key in result.__slots__:
                # noinspection PyBroadException
                try:
                    if key in UniversalTestcaseFactory.get_all_visible_headers():
                        setattr(result, key, UniversalTestcaseFactory.get_header_func(key)(dict_testcase[key]))
                    else:
                        setattr(result, key, dict_testcase[key])
                except:
                    logging.error("Failed parsing %s with value %s", key, dict_testcase[key])
        return result


class UniversalTestcaseFactory:
    """
    Universal Testcase Manager
    """

    def __init__(self, file: TextIO):
        """
        Store the whole Testcases in the csv file into memory
        """
        set_process_name("TestcaseManager")
        set_thread_name("Initialization")
        # Raw rows
        self.raw_data = []
        # Headers
        self.header = []
        csvreader = csv.reader(file)
        for row in csvreader:
            row = [column.strip() for column in row]
            self.raw_data.append(row)
        # First line should always be the title
        try:
            self.header = self.raw_data[0]
        except IndexError:
            logging.error("Testcase initialization received IndexError, csv file might be empty!")
            raise
        del self.raw_data[0]

        set_thread_name("HeaderCheckTestcaseName")
        # Testcase name generation
        if "testcase_name" not in self.header:
            logging.warning("Testcase name not found!"
                            " It is important to add a testcase_name in order to identify your testcases")
            self.header.append("testcase_name")
            for idx, row in enumerate(self.raw_data):
                row.append("auto_testcase_name_%d" % (idx + 1))

        set_thread_name("HeaderCheckUnidentifiedHeaders")
        self.ignored_headers = []
        for actual_header in self.header:
            if not self._is_legit_header(actual_header):
                logging.warning(f"Detected unidentified header: \"{actual_header}\", ignoring.")
                self.ignored_headers.append(actual_header)

        set_thread_name("HeaderCheckDuplicateHeaders")
        # Check duplicates
        header_check_set = set()
        for actual_header in self.header:
            if actual_header in header_check_set and actual_header not in self.ignored_headers:
                logging.error("Detected duplicate header: %s" % actual_header)
                raise RuntimeError("Detected duplicate header: %s" % actual_header)
            header_check_set.add(actual_header)

        set_thread_name("HeaderParseMapping")
        self.real_header_indexes = {}
        # Get idx of required testcase header in real header
        for header_name in self._get_all_legit_headers():
            header_idx = self._get_idx_of_header(header_name)
            self.real_header_indexes[header_name] = header_idx

        set_thread_name("TestcasePreParsing")
        raw_testcases = []

        # Process each line
        for idx, line in enumerate(self.raw_data):
            try:
                sub_result = self.__process(line, idx)
            except:
                logging.exception(f"Failed to process row index {idx}")
                raise
            raw_testcases.append(tuple(sub_result.values()))
        logging.debug(f"Processed {len(raw_testcases)} raw testcases")
        set_thread_name("FinalTestcaseStructureFormation")
        self.input_range_inferable = True
        self.output_range_inferable = True
        self.testcases = self._parse(raw_testcases)
        set_process_name()
        set_thread_name()

    @classmethod
    def _is_legit_header(cls, header_name: str) -> bool:
        if header_name not in UniversalTestcaseStructure.complete_headers:
            return False
        return True

    @classmethod
    def _has_equivalent_header(cls, header_name: str) -> Optional[bool]:
        if not cls._is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        if UniversalTestcaseStructure.complete_headers[header_name][1]:
            return True
        return False

    @classmethod
    def _get_equivalent_headers(cls, header_name: str) -> Optional[tuple]:
        if cls._has_equivalent_header(header_name):
            return UniversalTestcaseStructure.complete_headers[header_name][1]
        return ()

    @classmethod
    def _get_all_legit_headers(cls) -> Tuple[str]:
        return tuple(UniversalTestcaseStructure.complete_headers.keys())

    @classmethod
    def _has_default_value(cls, header_name: str) -> bool:
        if not cls._is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        if len(UniversalTestcaseStructure.complete_headers[header_name]) >= 3:
            return True
        return False

    @classmethod
    def _get_default_value(cls, header_name: str) -> Any:
        if cls._has_default_value(header_name):
            return UniversalTestcaseStructure.complete_headers[header_name][2]
        raise ValueError(f"Could not find default value for header {header_name}, required field missing!")

    @classmethod
    def _get_header_type(cls, header_name: str) -> FIELD_TYPES:
        if not cls._is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return UniversalTestcaseStructure.complete_headers[header_name][0]

    @classmethod
    def get_header_func(cls, header_name: str) -> Callable:
        if not cls._is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return UniversalTestcaseStructure.type_processing_func[cls._get_header_type(header_name)]

    @classmethod
    def get_all_visible_headers(cls):
        """All visible headers as csv titles"""
        return cls._get_all_legit_headers()

    def _get_idx_of_header(self, header_name, searched_tag=None) -> Optional[int]:
        """
        Get header position in actual header sequence
        :param header_name: string name of the header
        :param searched_tag: DO NOT USE
        :return: index of the header or its equivalent
        """
        result = None
        if header_name in self.header:
            result = self.header.index(header_name)
        else:
            equivalents = self._get_equivalent_headers(header_name)
            # Header not found, search for equivalent
            if not equivalents:
                return result
            # Initialize recursion variable
            if searched_tag is None:
                searched_tag = []
            for equivalent in equivalents:
                if equivalent not in searched_tag:
                    searched_tag.append(header_name)
                    result = self._get_idx_of_header(equivalent,
                                                     searched_tag)
                    if result is not None:
                        break
        return result

    def __process(self, row: list, row_index: int) -> Dict[str, Any]:
        # Initialize full testcase fields
        result = dict(zip(self._get_all_legit_headers(),
                          [PLACEHOLDER() for _ in self._get_all_legit_headers()]))
        changed, queue = self.__process_over_result(result, row, row_index)
        while changed:
            changed, queue = self.__process_over_result(result, row, row_index, queue)
        if queue:
            changed, queue = self.__process_over_result(result, row, row_index, queue, True)
        while changed:
            changed, queue = self.__process_over_result(result, row, row_index, queue)
        if queue:
            raise RuntimeError(f"Could not determine value of missing fields: {queue}")
        return result

    def __process_over_result(self, result, row, row_index, result_keys=None, apply_default=False):
        changed = False
        placeholder_queue = []
        keys = result_keys if result_keys else result
        for current_header_name in keys:
            column_value = result[current_header_name]
            # Parsed line
            if not isinstance(column_value, PLACEHOLDER):
                continue
            header_raw_idx = self.real_header_indexes[current_header_name]
            # Header not exist or value is empty, check for equivalent and default value
            if header_raw_idx is None or header_raw_idx >= len(row) or not row[header_raw_idx]:
                if self._has_equivalent_header(current_header_name) and not apply_default:
                    # Check for equivalent
                    resolved = False
                    value = None
                    for equivalent_header in self._get_equivalent_headers(current_header_name):
                        equivalent_header_value = result[equivalent_header]
                        if isinstance(equivalent_header_value, PLACEHOLDER):
                            # Equivalent not exist either, check for next equivalent
                            continue
                        else:
                            value = equivalent_header_value
                            resolved = True
                            break
                    if not resolved:
                        placeholder_queue.append(current_header_name)
                        continue
                elif self._has_default_value(current_header_name):
                    # Check for default value
                    changed = True
                    default_value = str(self._get_default_value(current_header_name))
                    result[current_header_name] = \
                        self.get_header_func(current_header_name)(default_value)
                    continue
                else:
                    placeholder_queue.append(current_header_name)
                    continue
            else:
                value = row[header_raw_idx]
            try:
                changed = True
                if isinstance(value, str):
                    result[current_header_name] = self.get_header_func(current_header_name)(value)
                else:
                    result[current_header_name] = value
            except:
                logging.exception(f"Failed to process header {current_header_name} of row {row_index} with "
                                  f"value {value}")
                raise
        return changed, placeholder_queue

    def get(self) -> Dict[int, set]:
        """
        Get all testcases
        :return:
        """
        return self.testcases

    # noinspection PyBroadException
    def _parse(self, testcases: list) -> Dict[int, set]:
        parsed_testcases = []
        testcase_names = set()
        # For testcase_count selector
        if 0 < get_global_storage().selected_testcase_count < len(testcases):
            logging.info("Selecting %d cases from all testcases" % get_global_storage().selected_testcase_count)
            all_indexes = random.sample(tuple(range(len(testcases))), k=get_global_storage().selected_testcase_count)
            testcases = [testcase for testcase_idx, testcase in enumerate(testcases) if testcase_idx in all_indexes]
        # Iterate through testcases, eliminate duplicates and do shape-range inference
        for testcase_idx, testcase in enumerate(testcases):
            # Construct Testcase Structure
            testcase_struct = UniversalTestcaseStructure()
            testcase_struct.original_line = self.raw_data[testcase_idx]
            testcase_struct.original_header = self.header
            headers = self._get_all_legit_headers()
            unidentified_headers = []
            for header_idx, header in enumerate(headers):
                if hasattr(testcase_struct, header):
                    setattr(testcase_struct, header, testcase[header_idx])
                else:
                    unidentified_headers.append(header)
            if unidentified_headers:
                raise KeyError(f"TestcaseManager header not match with UniversalTestcaseStructure, "
                               f"Report Bug to us: {unidentified_headers}")
            # Skip testcase if it is disabled
            if not testcase_struct.is_enabled:
                logging.debug("Testcase %s skipped" % testcase_struct.testcase_name)
                continue
            # Skip testcase if it is not in global testcase_name selector range
            if not self._check_testcase_name_selection(testcase_struct):
                continue
            # Skip testcase if it is not in global testcase_index selector range
            if not self._check_testcase_indexes_selection(testcase_idx, testcase_struct):
                continue
            # Skip testcase if it is not in global testcase_op_name selector range
            if not self._check_testcase_operator_selection(testcase_struct.op_name):
                continue
            ########
            # IMPORTANT !!! enabled testcases will now be recorded !!! IMPORTANT
            ########
            if testcase_struct.testcase_name in testcase_names:
                logging.warning("Detected duplicate testcase name: %s" % testcase_struct.testcase_name)
            testcase_names.add(testcase_struct.testcase_name)
            parsed_testcases.append(testcase_struct)
            set_thread_name(testcase_struct.testcase_name)
            try:
                if not self._check_other_params(testcase_struct):
                    continue
                if not self._parse_stc_input_dtypes(testcase_struct):
                    continue
                if not self._parse_stc_outputs(testcase_struct):
                    continue
                if not self._parse_stc_ori_outputs(testcase_struct):
                    continue
                if not self._parse_output_dtypes(testcase_struct):
                    continue
                self._check_dyn_inputs(testcase_struct)
                self._parse_dyn_input_dtypes(testcase_struct)
                self._check_const_inputs(testcase_struct)
                self._parse_dyn_outputs(testcase_struct)
                self._parse_dyn_ori_outputs(testcase_struct)
                self._parse_dyn_input_ranges(testcase_struct)
                self._parse_dyn_output_ranges(testcase_struct)
                if not self._stc_shape_size_check(testcase_struct):
                    continue
                if not self.__perform_shape_check(testcase_struct):
                    continue
                testcase_struct.network_name = get_networkname_by_testcase_name(testcase_struct.testcase_name)
            except:
                raise RuntimeError(f"Failed parsing testcase {testcase_struct.testcase_name}")
        # Construct final result
        testcase_group_dict = {}
        for parsed_testcase in parsed_testcases:
            hash_index = parsed_testcase.get_compilation_hash()
            testcase_group = testcase_group_dict.setdefault(hash_index, set())
            if parsed_testcase not in testcase_group:
                testcase_group.add(parsed_testcase)
            else:
                logging.warning("Duplicate testcase: %s" % parsed_testcase.testcase_name)
        return testcase_group_dict

    @staticmethod
    def _check_dyn_inputs(testcase_struct) -> NoReturn:
        if testcase_struct.dyn_inputs is None:
            testcase_struct.dyn_is_valid = False
            testcase_struct.dyn_fail_reason = "DYN_INPUT_MISSING"
            testcase_struct.dyn_inputs = testcase_struct.stc_inputs

    @staticmethod
    def _parse_dyn_outputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if isinstance(testcase_struct.dyn_outputs, str):
                    testcase_struct.dyn_outputs = do_shape_inference(testcase_struct.dyn_inputs,
                                                                     testcase_struct.dyn_outputs,
                                                                     testcase_struct.other_compilation_params)
                elif testcase_struct.dyn_outputs is None:
                    testcase_struct.dyn_outputs = tensorflow_inference(testcase_struct)
                return
            except:
                logging.warning("Dynamic output shape inference failure, use static output instead.")
        testcase_struct.dyn_outputs = testcase_struct.stc_outputs

    @staticmethod
    def _parse_dyn_ori_outputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if isinstance(testcase_struct.dyn_ori_outputs, str):
                    testcase_struct.dyn_ori_outputs = do_shape_inference(testcase_struct.dyn_ori_inputs,
                                                                         testcase_struct.dyn_ori_outputs,
                                                                         testcase_struct.other_compilation_params)
                elif testcase_struct.dyn_ori_outputs is None:
                    testcase_struct.dyn_ori_outputs = testcase_struct.dyn_outputs
                return
            except:
                logging.exception("Dynamic original output shape inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_ORI_OUTPUT_INFERENCE_FAILED"
        testcase_struct.dyn_ori_outputs = testcase_struct.stc_ori_outputs

    def _parse_dyn_output_ranges(self, testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if len(testcase_struct.dyn_output_ranges) == 0 or testcase_struct.dyn_output_ranges is None:
                    self.output_range_inferable = self.output_range_inferable and True
                    testcase_struct.dyn_output_ranges = shape_inference(testcase_struct.dyn_outputs, (), "RANGE")
                elif shape_inference(testcase_struct.dyn_outputs, (), "RANGE") == testcase_struct.dyn_output_ranges:
                    self.output_range_inferable = self.output_range_inferable and True
                else:
                    self.output_range_inferable = self.output_range_inferable and False
            except:
                logging.exception("Dynamic output range inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_OUTPUT_RANGE_INFERENCE_FAILED"

    @staticmethod
    def _parse_dyn_input_dtypes(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                testcase_struct.dyn_input_dtypes = tuple(parse_dtype(dtype)
                                                         for dtype in testcase_struct.dyn_input_dtypes)
                return
            except:
                logging.exception("Dynamic input dtypes parsing failure, turning dynamic profiling off")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_INPUT_DTYPES_INVALID"
        testcase_struct.dyn_input_dtypes = testcase_struct.stc_input_dtypes

    def _parse_dyn_input_ranges(self, testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if testcase_struct.dyn_input_ranges is None or len(testcase_struct.dyn_input_ranges) == 0:
                    testcase_struct.dyn_input_ranges = shape_inference(testcase_struct.dyn_inputs, (), "RANGE")
                    self.input_range_inferable = self.input_range_inferable and True
                elif shape_inference(testcase_struct.dyn_inputs, (), "RANGE") == testcase_struct.dyn_input_ranges:
                    self.input_range_inferable = self.input_range_inferable and True
                else:
                    self.input_range_inferable = self.input_range_inferable and False
            except:
                logging.exception("Dynamic input range inference failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "DYN_INPUT_RANGE_INFERENCE_FAILED"

    @staticmethod
    def _parse_stc_outputs(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if isinstance(testcase_struct.stc_outputs, str):
                testcase_struct.stc_outputs = do_shape_inference(testcase_struct.stc_inputs,
                                                                 testcase_struct.stc_outputs,
                                                                 testcase_struct.other_runtime_params)
            elif testcase_struct.stc_outputs is None:
                testcase_struct.stc_outputs = tensorflow_inference(testcase_struct)
        except:
            logging.exception("Static output shape inference failure")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_OUTPUT_INFERENCE_FAILED"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_stc_ori_outputs(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if isinstance(testcase_struct.stc_ori_outputs, str):
                testcase_struct.stc_ori_outputs = do_shape_inference(testcase_struct.stc_ori_inputs,
                                                                     testcase_struct.stc_ori_outputs,
                                                                     testcase_struct.other_runtime_params)
            elif testcase_struct.stc_ori_outputs is None:
                testcase_struct.stc_ori_outputs = tensorflow_inference(testcase_struct, True)
        except:
            logging.exception("Static original output shape inference failure")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_ORI_OUTPUT_INFERENCE_FAILED"
        return testcase_struct.is_valid

    @staticmethod
    def _check_testcase_name_selection(testcase_struct: UniversalTestcaseStructure) -> bool:
        if get_global_storage().selected_testcases:
            if testcase_struct.testcase_name \
                    in [s for s in get_global_storage().selected_testcases]:
                logging.info("Found selected testcase: %s" % testcase_struct.testcase_name)
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_indexes_selection(testcase_idx, testcase_struct: UniversalTestcaseStructure) -> bool:
        if get_global_storage().selected_testcase_indexes:
            if testcase_idx in get_global_storage().selected_testcase_indexes:
                logging.info("Found index %d of selected testcase: %s" % (testcase_idx,
                                                                          testcase_struct.testcase_name))
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_operator_selection(testcase_op_name) -> bool:
        if get_global_storage().selected_operators:
            if testcase_op_name in get_global_storage().selected_operators:
                return True
        else:
            return True
        return False

    @staticmethod
    def __perform_shape_check(testcase_struct: UniversalTestcaseStructure) -> bool:
        if testcase_struct.shape_check and testcase_struct.dyn_is_valid:
            # Static inputs num should never be greater than dynamic inputs num
            if len(testcase_struct.stc_inputs) > len(testcase_struct.dyn_inputs):
                logging.error(
                    "Testcase %s has been disabled because of dynamic-static tensor num difference"
                    % testcase_struct.testcase_name)
                testcase_struct.is_valid = False
                testcase_struct.fail_reason = "STC_TENSOR_GT_DYN_TENSOR"
            # Pre-DYN-STC-Shape check
            if (len(testcase_struct.const_input_indexes) + len(testcase_struct.dyn_inputs) ==
                    len(testcase_struct.stc_inputs)):
                # DYN-STC-Shape check
                const_input_offset = 0
                normalized_stc_shapes = eliminate_scalar_shapes(testcase_struct.stc_inputs)
                for shape_idx, dyn_shape in enumerate(eliminate_scalar_shapes(testcase_struct.dyn_inputs)):
                    if shape_idx in testcase_struct.const_input_indexes:
                        const_input_offset += 1
                    stc_shape = normalized_stc_shapes[shape_idx - const_input_offset]
                    # NoneType stc_shape and -2 dyn_shape should be ignored
                    if stc_shape is None or -2 in dyn_shape:
                        continue
                    if len(dyn_shape) != len(stc_shape):
                        logging.error(
                            "Testcase %s has been disabled because of dynamic-static shape length difference"
                            % testcase_struct.testcase_name)
                        testcase_struct.is_valid = False
                        testcase_struct.fail_reason = "DYN_STC_SHAPE_LEN"
                        break
                    if False in [dyn_dim == stc_dim or dyn_dim == -1
                                 for dyn_dim, stc_dim in zip(dyn_shape, stc_shape)]:
                        logging.error(
                            f"Testcase {testcase_struct.testcase_name} has been disabled because of "
                            f"dynamic-static shape dim difference: {dyn_shape} vs {stc_shape}")
                        testcase_struct.is_valid = False
                        testcase_struct.fail_reason = "DYN_STC_SHAPE_DIFF"
                        break
        return testcase_struct.is_valid

    @staticmethod
    def _stc_shape_size_check(testcase_struct: UniversalTestcaseStructure) -> bool:
        size_limitation = shape_size_check(testcase_struct.stc_inputs, testcase_struct.stc_outputs,
                                           testcase_struct.stc_input_dtypes, testcase_struct.output_dtypes)
        if size_limitation > 0:
            logging.error(f"Testcase {testcase_struct.testcase_name} is invalid because shape is out of bound: "
                          f"{size_limitation}")
            testcase_struct.is_valid = False
            testcase_struct.fail_reason = "STC_SHAPE_OUT_OF_BOUND"
        return testcase_struct.is_valid

    @staticmethod
    def _check_const_inputs(testcase_struct: UniversalTestcaseStructure) -> NoReturn:
        if testcase_struct.dyn_is_valid:
            # noinspection PyBroadException
            try:
                if len(testcase_struct.const_input_modes) == 0 and len(testcase_struct.const_input_indexes) > 0:
                    testcase_struct.const_input_modes = (None,)
            except:
                logging.exception("Const input modes or indexes parsing failure")
                testcase_struct.dyn_is_valid = False
                testcase_struct.dyn_fail_reason = "CONST_INPUT_INVALID"

    @staticmethod
    def _check_other_params(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            for param in testcase_struct.other_compilation_params:
                if param not in testcase_struct.other_runtime_params:
                    testcase_struct.other_runtime_params[param] = testcase_struct.other_compilation_params[param]
                elif param in testcase_struct.other_runtime_params and \
                        testcase_struct.other_runtime_params[param] != testcase_struct.other_compilation_params[param]:
                    logging.warning(f"Compilation param {param} conflicts with which in runtime params: "
                                    f"{testcase_struct.other_runtime_params[param]} vs "
                                    f"{testcase_struct.other_compilation_params[param]}")
        except:
            logging.exception("Compilation or Runtime params parsing failure")
            testcase_struct.dyn_is_valid = False
            testcase_struct.is_valid = False
            testcase_struct.dyn_fail_reason = testcase_struct.fail_reason = "OTHER_PARAMS_INVALID"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_stc_input_dtypes(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            if testcase_struct.is_valid:
                testcase_struct.stc_input_dtypes = tuple(parse_dtype(dtype)
                                                         for dtype in testcase_struct.stc_input_dtypes)
        except:
            logging.exception("Static input dtypes parsing failure")
            testcase_struct.is_valid = False
            testcase_struct.dyn_is_valid = False
            testcase_struct.fail_reason = "STC_INPUT_DTYPES_INVALID"
        return testcase_struct.is_valid

    @staticmethod
    def _parse_output_dtypes(testcase_struct: UniversalTestcaseStructure) -> bool:
        # noinspection PyBroadException
        try:
            testcase_struct.output_dtypes = tuple(parse_dtype(dtype) for dtype in testcase_struct.output_dtypes)
        except:
            logging.exception("Output dtypes parsing failure")
            testcase_struct.dyn_is_valid = False
            testcase_struct.is_valid = False
            testcase_struct.dyn_fail_reason = testcase_struct.fail_reason = "OUTPUT_DTYPES_INVALID"
        return testcase_struct.is_valid


def do_shape_inference(inputs: tuple, outputs: str, args: dict) -> tuple:
    """
    Shape inference
    :param inputs:
    :param outputs:
    :param args:
    :return:
    """
    if None in inputs:
        raise ValueError("Automatic inference doesn't support None input")
    if outputs in ("ELEWISE",):
        return shape_inference(inputs, (1, None), outputs)
    if outputs in ("REDUCE",):
        if "axes" in args:
            axes = args["axes"]
        elif "axis" in args:
            axes = args["axis"]
        else:
            axes = None
        return shape_inference(inputs, (axes, 1, None), outputs)
    elif "ELEWISE" in outputs:
        try:
            args = eval(outputs[7:])
        except:
            raise ValueError("Unable to parse shape inference args from %s" % outputs)
        else:
            return shape_inference(inputs, args, "ELEWISE")
    elif "REDUCE" in outputs:
        try:
            _args = eval(outputs[6:])
            if "axes" in args:
                args = (args["axes"], *_args[1:])
            elif "axis" in args:
                args = (args["axis"], *_args[1:])
            else:
                args = _args
        except:
            raise ValueError("Unable to parse shape inference args from %s" % outputs)
        else:
            return shape_inference(inputs, args, "REDUCE")
    else:
        raise ValueError("Invalid shape inference value %s" % outputs)


def shape_size_check(shapes: tuple, shapes_out: tuple, dtypes: tuple, dtypes_out: tuple) -> int:
    """
    Check shape product size, return None if shape size out of bound
    :param dtypes_out:
    :param dtypes:
    :param shapes_out:
    :param shapes:
    :return:
    """
    shapes = eliminate_scalar_shapes(shapes)
    shapes_out = eliminate_scalar_shapes(shapes_out)
    total_shape_product_value = 0
    for idx, shape in enumerate(shapes):
        if shape is not None:
            shape_size = shape_product(shape)
            total_shape_product_value += shape_size * get_dtype_width(get(dtypes, idx))

    for idx, shape in enumerate(shapes_out):
        if shape is not None:
            shape_size = shape_product(shape)
            total_shape_product_value += shape_size * get_dtype_width(get(dtypes_out, idx))

    if total_shape_product_value > get_global_storage().DAVINCI_HBM_SIZE_LIMIT:
        return get_global_storage().DAVINCI_HBM_SIZE_LIMIT
    return 0
