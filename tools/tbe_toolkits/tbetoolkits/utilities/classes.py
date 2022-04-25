#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious Utility Classes
"""
# Standard Packages
import json
import pathlib
from enum import auto
from enum import Enum
from typing import Optional

# Third-party Packages
import tbetoolkits


class PMU_TITLES:
    """
    TITLES FOR PMU
    """
    DEFAULT_TITLES = ("TOTAL",
                      "VECTOR",
                      "CUBE",
                      "SCALAR",
                      "MTE1",
                      "MTE2",
                      "MTE3",
                      "ICACHE_REQ",
                      "ICACHE_MISS",
                      "L2_REQ",
                      "L2_HIT",
                      "DIR_HIT",
                      "L2_VICTIM",
                      "READ_REQ",
                      "WRITE_REQ",
                      "ALLOC_REQ",
                      "READ_HIT_L2_FORWARD")
    ADVANCED_TITLES = ("TOTAL",
                       "BANKGROUP_CFLT",
                       "BANK_CFLT",
                       "RESC_CFLT",
                       "MTE1_IQFULL",
                       "MTE2_IQFULL",
                       "MTE3_IQFULL",
                       "CUBE_IQFULL",
                       "VEC_IQFULL",
                       "L2_REQ",
                       "L2_HIT",
                       "DIR_HIT",
                       "L2_VICTIM",
                       "READ_REQ",
                       "WRITE_REQ",
                       "ALLOC_REQ",
                       "READ_HIT_L2_FORWARD")


class MODE(Enum):
    """Model Type"""
    TESTCASE_UPLOAD = auto()
    ASCEND_ONBOARD = auto()
    ASCEND_CAMODEL = auto()
    ASCEND_PEMMODEL = auto()
    ASCEND_ESLMODEL = auto()
    GPU_TENSORFLOW = auto()
    GPU_PYTORCH = auto()

    def is_model(self) -> Optional[str]:
        if self in [MODE.ASCEND_ESLMODEL, MODE.ASCEND_CAMODEL, MODE.ASCEND_PEMMODEL]:
            return self.name.split("_")[-1]
        return None

    def is_gpu(self) -> Optional[str]:
        if self in [MODE.GPU_TENSORFLOW, MODE.GPU_PYTORCH]:
            return self.name.split("_")[-1]
        return None


class DUMP_LEVEL(Enum):
    """Dump data level"""
    NO = 0b000
    INPUT = 0b100
    OUTPUT = 0b010
    INOUT = 0b110
    GOLDEN = 0b001
    INGOLD = 0b101
    OUTGOLD = 0b011
    FULL = 0b111


class PMU_MODE(Enum):
    """PMU Mode"""
    DEFAULT = auto()
    ADVANCED = auto()
    L2 = auto()


class SWITCHES:
    """
    Control Panel
    """

    __slots__ = [
        "mode",
        "input_file_name",
        "output_file_name",
        "single_case_debugging",
        "logging_to_file",
        "single_testcase_log_mode",
        "device_platform",
        "core_type",
        "soc_spec_override",
        "process_per_device",
        "parallel_fatbin",
        "dyn_switches",
        "stc_switches",
        "cst_switches",
        "bin_switches",
        "PMU",
        "PMU_MODE",
        "do_precision_test",
        "dump_mode",
        "int64_shape_mode",
        "device_count",
        "device_blacklist",
        "run_time",
        "tiling_run_time",
        "perf_threshold",
        "kernel_meta",
        "model_update_configs",
        "model_target_block_dim",
        "summary_print",
        "DAVINCI_HBM_SIZE_LIMIT",
        "PMU_RETURN_SIZE",
        "selected_testcases",
        "selected_testcase_indexes",
        "selected_testcase_count",
        "selected_operators",
        "testcase_server",
        "preserve_original_csv"
    ]

    def __init__(self):
        self.mode = MODE.ASCEND_ONBOARD
        self.input_file_name = None
        self.output_file_name = None
        self.single_case_debugging: bool = False
        self.logging_to_file: bool = False
        self.single_testcase_log_mode = False
        self.device_platform = "AUTO"
        self.core_type = "AiCore"
        # Override soc spec such as CORE_NUM
        self.soc_spec_override = {}
        self.process_per_device = None
        self.parallel_fatbin = None
        self.dyn_switches: OPTestSwitch = OPTestSwitch("Dynamic shape", True,
                                                       True,
                                                       True,
                                                       True)
        self.stc_switches: OPTestSwitch = OPTestSwitch("Static shape", True,
                                                       True,
                                                       True,
                                                       True)
        self.cst_switches: OPTestSwitch = OPTestSwitch("Const shape", False,
                                                       True,
                                                       True,
                                                       True)
        self.bin_switches: OPTestSwitch = OPTestSwitch("Binary release", False,
                                                       True,
                                                       True,
                                                       True)
        self.PMU = False
        self.PMU_MODE = PMU_MODE.DEFAULT
        self.do_precision_test = True
        self.dump_mode = DUMP_LEVEL.NO
        self.int64_shape_mode = False
        self.device_count = -1
        self.device_blacklist = []
        self.run_time = 3
        self.tiling_run_time = 3
        self.perf_threshold = (1.0, 10)
        # Hidden switches
        self.kernel_meta = pathlib.Path("./kernel_meta/")
        self.model_update_configs = True
        self.model_target_block_dim = (0,)
        self.summary_print = True
        # Constants
        self.DAVINCI_HBM_SIZE_LIMIT = 1024 * 1024 * 1024 * 30  # 1024B * 1024KB * 1024MB * 30GB
        self.PMU_RETURN_SIZE = 17
        # Testcases
        self.selected_testcases = []
        self.selected_testcase_indexes = []
        self.selected_testcase_count = -1
        self.selected_operators = None
        self.testcase_server = None
        self.preserve_original_csv = False


class OPTestSwitch:
    """
    e.g. dynamic_shape, static_shape
    """

    def __init__(self, name, switch, realtime_compilation, profiling, online_profiling):
        self.name = name
        self.enabled = switch
        self.realtime = realtime_compilation
        self.prof = profiling
        self.rts_prof = online_profiling

    def __str__(self):
        return "%s Status: %s, %s, %s, %s" % (self.name,
                                              "ENABLED" if self.enabled else "DISABLED",
                                              "TE_COMPILE" if self.realtime else "MANUAL_COMPILE",
                                              "ONLINE" if self.prof else "OFFLINE",
                                              "PROFILE" if str(self.rts_prof) else "RUN_ONLY")

    def get_prof(self):
        """
        Get profiling switch based on model mode
        :return:
        """
        return self.prof


class DynamicCompilationResult:
    """For Dynamic Compilation"""

    def __init__(self):
        # Independent
        self.kernel_name: Optional[str] = None
        self.tiling_result: Optional[DynamicOpTilingResult] = None
        # Dependent
        self.compile_result: Optional[str] = None
        self.compile_time: Optional[str] = None
        self.compile_info: Optional[dict] = None
        self.tiling_op_type: Optional[str] = None
        self.func_params: Optional[tuple] = None
        self.sch_count: Optional[str] = None
        self.obj_size: Optional[str] = None
        self.op_pattern: Optional[str] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        self.kernel_name = value
        self.compile_result = value
        self.compile_time = value
        self.compile_info = {}
        self.tiling_op_type = value
        self.func_params = None
        self.sch_count = value
        self.obj_size = value
        self.op_pattern = value

    def standard_set(self,
                     a: Optional[str], b: Optional[str], c: Optional[dict], d: Optional[str],
                     e: Optional[tuple], f: Optional[str], g: Optional[str], h: Optional[str]):
        """Set standard value"""
        self.compile_result = a
        self.compile_time = b
        self.compile_info = c
        self.tiling_op_type = d
        self.func_params = e
        self.sch_count = f
        self.obj_size = g
        self.op_pattern = h

    def write_json(self, path: Optional[str]):
        """Write compile info json"""
        json_parsed = {"compile_info": self.compile_info, "tiling_op_type": self.tiling_op_type,
                       "dyn_func_params": self.func_params}
        with open(pathlib.Path(path, "%s.tbetoolkits" % self.kernel_name),
                  "w+", encoding="UTF-8") as json_file:
            json_file.write(json.dumps(json_parsed, indent=4))

    def standard_get(self):
        """Get standard value"""
        return (self.compile_result,
                self.compile_time,
                self.compile_info,
                self.tiling_op_type,
                self.func_params,
                self.sch_count,
                self.obj_size,
                self.op_pattern)

    def apply(self, testcase: "tbetoolkits.UniversalTestcaseStructure"):
        """Apply dynamic result to testcase"""
        testcase.dyn_kernel_name = self.kernel_name
        testcase.dyn_compile_result = self.compile_result
        testcase.dyn_compile_time = self.compile_time
        testcase.dyn_compile_info = self.compile_info
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.dyn_sch_count = self.sch_count
        testcase.dyn_obj_size = self.obj_size
        testcase.dyn_op_pattern = self.op_pattern
        testcase.dyn_block_dim = self.tiling_result.dyn_block_dim
        testcase.dyn_tiling_key = self.tiling_result.dyn_tiling_key
        testcase.dyn_tiling_data = self.tiling_result.dyn_tiling_data
        testcase.dyn_workspaces = self.tiling_result.dyn_workspaces
        testcase.dyn_tiling_time = self.tiling_result.dyn_tiling_time
        testcase.dyn_kernel_size = self.tiling_result.dyn_kernel_size
        testcase.ready_for_profile += 1


class StaticCompilationResult:
    """For Static Compilation"""

    def __init__(self):
        # Independent
        self.stc_kernel_name: Optional[str] = None
        # Dependent
        self.stc_compile_result: Optional[str] = None
        self.compile_time: Optional[str] = None
        self.stc_block_dim: Optional[int] = None
        self.stc_workspaces: Optional[tuple] = None
        self.rl_query_result: Optional[str] = None
        self.stc_op_pattern: Optional[str] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        self.stc_kernel_name = value
        self.stc_compile_result = value
        self.compile_time = value
        self.stc_block_dim = 0
        self.stc_workspaces = ()
        self.rl_query_result = value
        self.stc_op_pattern = value

    def standard_set(self,
                     a: Optional[str], b: Optional[str], c: Optional[int],
                     d: Optional[tuple], e: Optional[str], f: Optional[str]):
        """Set standard value"""
        self.stc_compile_result = a
        self.compile_time = b
        self.stc_block_dim = c
        self.stc_workspaces = d
        self.rl_query_result = e
        self.stc_op_pattern = f

    def standard_get(self):
        """Get standard value"""
        return (self.stc_compile_result,
                self.compile_time,
                self.stc_block_dim,
                self.stc_workspaces,
                self.rl_query_result,
                self.stc_op_pattern)

    def apply(self, testcase: "tbetoolkits.UniversalTestcaseStructure"):
        """Apply result to testcase"""
        testcase.stc_kernel_name = self.stc_kernel_name
        testcase.stc_compile_result = self.stc_compile_result
        testcase.stc_compile_time = self.compile_time
        testcase.stc_block_dim = self.stc_block_dim
        testcase.stc_workspaces = self.stc_workspaces
        testcase.stc_rl_query_result = self.rl_query_result
        testcase.stc_op_pattern = self.stc_op_pattern
        testcase.ready_for_profile += 1


class ConstCompilationResult:
    """For const Compilation"""

    def __init__(self):
        # Independent
        self.cst_kernel_name: Optional[str] = None
        # Dependent
        self.cst_compile_result: Optional[str] = None
        self.compile_time: Optional[str] = None
        self.cst_block_dim: Optional[int] = None
        self.cst_workspaces: Optional[tuple] = None
        self.cst_func_params: Optional[tuple] = None
        self.op_pattern: Optional[tuple] = None
        self.cst_rl_status: Optional[str] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        self.cst_kernel_name = value
        self.cst_compile_result = value
        self.compile_time = value
        self.cst_block_dim = 0
        self.cst_workspaces = ()
        self.cst_func_params = None
        self.op_pattern = value
        self.cst_rl_status = value

    def standard_set(self,
                     a: Optional[str], b: Optional[str], c: Optional[int],
                     d: Optional[tuple], e: Optional[tuple], f: Optional[str], g: Optional[str]):
        """Set standard value"""
        self.cst_compile_result = a
        self.compile_time = b
        self.cst_block_dim = c
        self.cst_workspaces = d
        self.cst_func_params = e
        self.op_pattern = f
        self.cst_rl_status = g

    def standard_get(self):
        """Get standard value"""
        return (self.cst_compile_result,
                self.compile_time,
                self.cst_block_dim,
                self.cst_workspaces,
                self.cst_func_params,
                self.op_pattern,
                self.cst_rl_status)

    def apply(self, testcase: "tbetoolkits.UniversalTestcaseStructure"):
        """Apply result to testcase"""
        testcase.cst_kernel_name = self.cst_kernel_name
        testcase.cst_compile_result = self.cst_compile_result
        testcase.cst_compile_time = self.compile_time
        testcase.cst_block_dim = self.cst_block_dim
        testcase.cst_workspaces = self.cst_workspaces
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.cst_func_params
        testcase.cst_op_pattern = self.op_pattern
        testcase.cst_rl_status = self.cst_rl_status
        testcase.ready_for_profile += 1

    def write_json(self, path: Optional[str]):
        """Write compile info json"""
        json_parsed = {"cst_func_params": self.cst_func_params, "cst_rl_status": self.cst_rl_status}
        with open(pathlib.Path(path, "%s.tbetoolkits" % self.cst_kernel_name),
                  "w+", encoding="UTF-8") as json_file:
            json_file.write(json.dumps(json_parsed, indent=4))


class BinaryCompilationResult:
    """For Binary Compilation"""

    def __init__(self):
        # Independent
        self.kernel_name: Optional[str] = None
        self.tiling_result: Optional[BinaryOpTilingResult] = None
        # Dependent
        self.compile_result: Optional[str] = None
        self.compile_time: Optional[str] = None
        self.compile_info: Optional[dict] = None
        self.tiling_op_type: Optional[str] = None
        self.func_params: Optional[tuple] = None
        self.sch_count: Optional[str] = None
        self.obj_size: Optional[str] = None

    def all_set(self, value: Optional[str]):
        """Set all values"""
        self.kernel_name = value
        self.compile_result = value
        self.compile_time = value
        self.compile_info = {}
        self.tiling_op_type = value
        self.func_params = None
        self.sch_count = value
        self.obj_size = value

    def standard_set(self,
                     a: Optional[str], b: Optional[str], c: Optional[dict],
                     d: Optional[str], e: Optional[tuple], f: Optional[str], g: Optional[str]):
        """Set standard value"""
        self.compile_result = a
        self.compile_time = b
        self.compile_info = c
        self.tiling_op_type = d
        self.func_params = e
        self.sch_count = f
        self.obj_size = g

    def write_json(self, path: Optional[str]):
        """Write compile info json"""
        json_parsed = {"compile_info": self.compile_info, "tiling_op_type": self.tiling_op_type,
                       "bin_func_params": self.func_params}
        with open(pathlib.Path(path, "%s.tbetoolkits" % self.kernel_name),
                  "w+", encoding="UTF-8") as json_file:
            json_file.write(json.dumps(json_parsed, indent=4))

    def standard_get(self):
        """Get standard value"""
        return (self.compile_result,
                self.compile_time,
                self.compile_info,
                self.tiling_op_type,
                self.func_params,
                self.sch_count,
                self.obj_size)

    def apply(self, testcase: "tbetoolkits.UniversalTestcaseStructure"):
        """Apply result to testcase"""
        testcase.bin_kernel_name = self.kernel_name
        testcase.bin_compile_result = self.compile_result
        testcase.bin_compile_time = self.compile_time
        testcase.bin_compile_info = self.compile_info
        if testcase.dyn_func_params is None:
            testcase.dyn_func_params = self.func_params
        testcase.bin_sch_count = self.sch_count
        testcase.bin_obj_size = self.obj_size
        testcase.bin_block_dim = self.tiling_result.bin_block_dim
        testcase.bin_tiling_key = self.tiling_result.bin_tiling_key
        testcase.bin_tiling_data = self.tiling_result.bin_tiling_data
        testcase.bin_workspaces = self.tiling_result.bin_workspaces
        testcase.bin_tiling_time = self.tiling_result.bin_tiling_time
        testcase.bin_kernel_size = self.tiling_result.bin_kernel_size
        testcase.ready_for_profile += 1


class DynamicOpTilingResult:
    """For Dynamic Op-tiling"""

    def __init__(self):
        self.dyn_block_dim: Optional[int] = None
        self.dyn_tiling_key: Optional[int] = None
        self.dyn_tiling_data: Optional[bytes] = None
        self.dyn_workspaces: Optional[tuple] = None
        self.dyn_tiling_time: Optional[str] = None
        self.dyn_kernel_size: Optional[str] = None

    def standard_set(self,
                     a: Optional[str], b: Optional[int], c: Optional[bytes],
                     d: Optional[tuple], e: Optional[str], f: Optional[str]):
        """Set standard value"""
        self.dyn_block_dim = a
        self.dyn_tiling_key = b
        self.dyn_tiling_data = c
        self.dyn_workspaces = d
        self.dyn_tiling_time = e
        self.dyn_kernel_size = f

    def all_set(self, value: Optional[str]):
        """Set standard value"""
        self.dyn_block_dim = 0
        self.dyn_tiling_key = -1
        self.dyn_tiling_data = b""
        self.dyn_workspaces = ()
        self.dyn_tiling_time = value
        self.dyn_kernel_size = value


class BinaryOpTilingResult:
    """For Binary Op-tiling"""

    def __init__(self):
        pass
        self.bin_block_dim: Optional[int] = None
        pass
        self.bin_tiling_key: Optional[int] = None
        pass
        self.bin_tiling_data: Optional[bytes] = None
        pass
        self.bin_workspaces: Optional[tuple] = None
        pass
        self.bin_tiling_time: Optional[str] = None
        self.bin_kernel_size: Optional[str] = None

    def standard_set(self,
                     a: Optional[str], b: Optional[int], c: Optional[bytes],
                     d: Optional[tuple], e: Optional[str], f: Optional[str]):
        """Set standard value"""
        self.bin_block_dim = a
        self.bin_tiling_key = b
        self.bin_tiling_data = c
        self.bin_workspaces = d
        self.bin_tiling_time = e
        self.bin_kernel_size = f

    def all_set(self, value: Optional[str]):
        """Set standard value"""
        self.bin_block_dim = 0
        self.bin_tiling_key = -1
        self.bin_tiling_data = b""
        self.bin_workspaces = ()
        self.bin_tiling_time = value
        self.bin_kernel_size = value
