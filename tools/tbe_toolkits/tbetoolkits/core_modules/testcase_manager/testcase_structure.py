#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Testcase Structure
"""
# Standard Packages
import hashlib
import logging
from typing import Any, Callable, Dict, Optional, Tuple

# Third-Party Packages
import tbetoolkits
from .field_types import FIELD_TYPES
from .field_parser import *


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
                 "torch_func",
                 "stc_const_input_indexes",
                 "kb_pid")
    identity_headers: Dict[str, tuple] = {
        "testcase_name": (FIELD_TYPES.STRING, None),  # Required
        "network_name": (FIELD_TYPES.STRING, None, None),
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
    special_property_headers: Dict[str, tuple] = {
        "dyn_input_as_list_distribution": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "stc_input_as_list_distribution": (FIELD_TYPES.INT_CONTAINER, None, ()),
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
        **non_platform_static_property_headers, **static_property_headers, **dynamic_property_headers,
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
        self.stc_const_input_indexes: Optional[tuple] = None
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
        self.kb_pid: int = None

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
            self.const_input_indexes,
            self.const_input_modes,
            self.manual_dyn_build_config,
            self.other_compilation_params,
            self.random_buff)
        return hash(str(compilation_params))

    def as_dict(self) -> dict:
        titles = tuple(self.identity_headers.keys()) + tuple(self.property_headers.keys())
        values = tuple(getattr(self, title) for title in titles)
        return dict(zip(titles, values))

    def get_hash(self) -> bytes:
        """
        GPU Testcase uniqueness
        """
        values = tuple(getattr(self, title) for title in tuple(self.non_platform_static_property_headers.keys()))
        result_dict = dict(zip(tuple(self.non_platform_static_property_headers.keys()), values))
        return hashlib.blake2b(str(result_dict).encode('UTF-8')).digest()

    def __hash__(self):
        return hash(self.testcase_name)

    def ready(self) -> bool:
        """Ready for profile"""
        if self.ready_for_profile == 4:
            return True
        return False

    @classmethod
    def is_legit_header(cls, header_name: str) -> bool:
        return header_name in cls.complete_headers

    @classmethod
    def has_equivalent_header(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return True if cls.complete_headers[header_name][1] else False

    @classmethod
    def get_equivalent_headers(cls, header_name: str) -> Optional[tuple]:
        if cls.has_equivalent_header(header_name):
            return cls.complete_headers[header_name][1]
        return ()

    @classmethod
    def get_all_legit_headers(cls) -> Tuple[str]:
        return tuple(cls.complete_headers.keys())

    @classmethod
    def has_default_value(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return len(cls.complete_headers[header_name]) >= 3

    @classmethod
    def get_default_value(cls, header_name: str) -> Any:
        if cls.has_default_value(header_name):
            return cls.complete_headers[header_name][2]
        raise ValueError(f"Could not find default value for header {header_name}, required field missing!")

    @classmethod
    def get_header_type(cls, header_name: str) -> FIELD_TYPES:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return cls.complete_headers[header_name][0]

    @classmethod
    def get_header_func(cls, header_name: str) -> Callable:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return cls.type_processing_func[cls.get_header_type(header_name)]

    @classmethod
    def get_all_visible_headers(cls):
        """All visible headers as csv titles"""
        return cls.get_all_legit_headers()

    @staticmethod
    def from_dict(dict_testcase: dict):
        result = UniversalTestcaseStructure()
        for key in dict_testcase:
            if key in result.__slots__:
                # noinspection PyBroadException
                try:
                    if key in result.get_all_visible_headers():
                        setattr(result, key, result.get_header_func(key)(dict_testcase[key]))
                    else:
                        setattr(result, key, dict_testcase[key])
                except:
                    logging.error("Failed parsing %s with value %s", key, dict_testcase[key])
        return result