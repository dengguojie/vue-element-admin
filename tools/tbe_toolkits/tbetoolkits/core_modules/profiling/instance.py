#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Main Sequence for dynamic shape profiling
"""
# Standard Packages
import io
import csv
import copy
import time
import zipfile
import random
import logging
import multiprocessing
from enum import Enum
from enum import auto
from typing import Set
from typing import Optional
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable

# Third-Party Packages
from .profiling import profile_process
from .profiling import ProfilingReturnStructure
from .compilation import compilation_process

from .. import dsmi
from .. import downloader

from ..tbe_multiprocessing.pool import SimpleCommandProcess
from ..tbe_multiprocessing.pool import get_cpu_count

from ..testcase_manager import UniversalTestcaseFactory
from ..testcase_manager import UniversalTestcaseStructure

from ...utilities import VERSION
from ...utilities import get_global_storage
from ...utilities import table_print
from ...utilities import PMU_MODE
from ...utilities import PMU_TITLES
from ...utilities import tuple_flatten
from ...utilities import DynamicCompilationResult
from ...utilities import StaticCompilationResult
from ...utilities import ConstCompilationResult
from ...utilities import BinaryCompilationResult
from ...utilities import DynamicOpTilingResult
from ...utilities import BinaryOpTilingResult


class SubTaskType(Enum):
    COMPILE = auto()
    PROFILE = auto()


class Task:
    def __init__(self, group_id: int, testcase_struct: UniversalTestcaseStructure):
        self.group_id = group_id
        self.testcase_struct = testcase_struct


class SubTask:
    def __init__(self, task: Optional[Task], sub_task_type: SubTaskType, func: Callable, params: tuple):
        self.task = task
        self.type = sub_task_type
        self.func = func
        self.params = params
        self.process: Optional[SimpleCommandProcess] = None

    def send_to_proc(self, process: SimpleCommandProcess):
        process.send_action(self.func, self.params, {})
        self.process = process


class ProfilingInstance:
    """
    Dynamic Shape Profiling Interface
    """
    print_time = 10

    def __init__(self):
        self.dsmi = None
        # Switches Storage
        self.switches = get_global_storage()
        # gdb friendly mode
        self.debug_mode = self.switches.single_case_debugging
        # Testcases
        self.testcases: Optional[Dict[int, Set[UniversalTestcaseStructure]]] = None
        self.flatten_testcases: Optional[Tuple[UniversalTestcaseStructure]] = None
        self.titles = None
        # Test Result CSV Storage
        self.csv_writer = None
        self.dyn_pmu_titles = ()
        self.stc_pmu_titles = ()
        self.cst_pmu_titles = ()
        self.result_titles = ()
        self.input_path: str = ""
        self.result_path: str = ""
        # Multiprocessing
        self.mp_context = multiprocessing.get_context("forkserver")
        self.manager = None
        self.device_locks = ()
        self.device_to_processes: Dict[int, Tuple[SimpleCommandProcess, ...]] = {}
        self.process_to_device: Dict[SimpleCommandProcess, int] = {}
        # Initialized tasks without device
        self.waiting_tasks: List[Task, ...] = []
        self.total_tasks_count: int = 0
        self.completed_task_count: int = 0
        # Tasks executing
        self.process_to_subtask: Dict[SimpleCommandProcess, SubTask] = {}
        # Tasks waiting for execution
        self.device_subtasks: Dict[int, List[SubTask, ...]] = {}
        self.last_print_timestamp = time.time() - 20
        self.start_timestamp = time.time()

    def profile(self):
        logging.info("Preparing TBEToolkits Ascend Profiling Infrastructure...")
        logging.info("Mode: %s", self.switches.mode.name)
        if self.switches.mode.is_gpu():
            raise RuntimeError(f"Please do not call Ascend Profiling Instance in {self.switches.mode.name} Mode")
        if self.switches.mode.is_model():
            if self.switches.device_platform == "AUTO":
                raise RuntimeError(f"Please specify your platform type in {self.switches.mode.name} mode")
            self.dsmi = None
            self.switches.device_count = 1
        elif self.switches.mode != self.switches.mode.TESTCASE_UPLOAD:
            if self.switches.device_count == -1 or self.switches.device_platform == "AUTO":
                self.dsmi = dsmi.DSMIInterface()
            # Call DSMI Interface
            if self.switches.device_count == -1:
                logging.info("Ascend SoC Device count unknown, get device count through DSMI...")
                self.switches.device_count = self.dsmi.get_device_count()
            if self.switches.device_platform == "AUTO":
                logging.info("Ascend SoC Device platform unknown, get chip type of device 0 through DSMI...")
                self.switches.device_platform = self.dsmi.get_chip_info(0).get_complete_platform()
            if self.switches.device_count <= 0:
                raise RuntimeError(f"Ascend SoC Device count is invalid: {self.switches.device_count}")
        # Prepare titles
        self._prepare_result_titles()
        # Check result csv
        self.input_path = self.switches.input_file_name
        self.result_path = self.switches.output_file_name
        if self.input_path is None:
            raise RuntimeError("Please specify input csv file path")
        if self.result_path is None:
            split_input_path = self.input_path.split(".")
            split_input_path[-2] += "_result"
            self.result_path = '.'.join(split_input_path)
        if not self.result_path.endswith('.csv'):
            self.result_path += '.csv'
        # Download models
        if self.switches.mode.is_model():
            downloader.get_models(self.switches.device_platform, self.switches.mode)
        logging.info(f"Ascend Device Platform: {self.switches.device_platform}")
        logging.info(f"Ascend Device Core Type: {self.switches.core_type}")
        logging.info(f"Ascend Device Device Count: {self.switches.device_count}")
        logging.info(f"Ascend Device PMU switch: {self.switches.PMU}")
        logging.info(f"Ascend Device PMU Mode: {self.switches.PMU_MODE.name}")
        logging.info(f"result_path: {self.result_path}")
        logging.info(str(self.switches.dyn_switches))
        logging.info(str(self.switches.stc_switches))
        logging.info(str(self.switches.cst_switches))
        logging.info(str(self.switches.bin_switches))
        logging.info(f"Precision Comparison: {self.switches.do_precision_test}")
        # Prepare device locks
        self.device_locks = self.initialize_device_lock()
        # Print Device blacklist info
        for blacklist_device_id in self.switches.device_blacklist:
            logging.info(f"Device {blacklist_device_id} has been blacklisted, removing...")
        # Prepare testcases
        logging.info("Parsing testcases...")
        self._prepare_testcases(self.input_path)
        logging.info(f"Case num: {len(tuple_flatten(tuple(self.testcases.values())))}")
        # Prepare SubProcesses
        logging.info("Preparing Task Executors...")
        self._prepare_processes()
        logging.info("Initializing Task Executors...")
        self._initialize_processes()
        # Write csv title
        logging.info("Initialize output csv file...")
        self.init_flush(self.result_path)
        self.flush((*self.titles, "<-INPUT-|-RESULT->", *self.result_titles))
        logging.info("Preparing compilation tasks")
        self._prepare_tasks()
        logging.info("Received compilation tasks: %d" % len(self.waiting_tasks))
        # Loop check compiling and add to profiling pool
        while True:
            self._update_processes()
            self._push_tasks()
            if self.total_tasks_count == self.completed_task_count:
                logging.info(f"TBEToolkits {self.switches.mode} Profiling complete")
                break
        # Close all processes
        for proc in self.process_to_device:
            proc.close()

    def init_flush(self, file_path: str):
        self.csv_writer = csv.writer(open(file_path, newline='', mode='w+'))

    def flush(self, row: tuple):
        self.csv_writer.writerow(row)

    def initialize_device_lock(self) -> tuple:
        self.manager = self.mp_context.Manager()
        return tuple(self.manager.Lock() if n not in self.switches.device_blacklist else None
                     for n in range(self.switches.device_count))

    @staticmethod
    def __get_process_stage_info(proc: SimpleCommandProcess) -> str:
        return proc.data.get("stage")

    def _prepare_testcases(self, testcase_path: str):
        if self.debug_mode:
            logging.info("Entering single testcase debugging mode...")
        testcases: Dict[int, set] = {}
        try:
            with zipfile.ZipFile(testcase_path) as zipped_file:
                logging.info("Reading zipped testcases...")
                test_result = zipped_file.testzip()
                if test_result:
                    raise RuntimeError(f"Zipfile corrupted on file: {test_result}")
                all_files = zipped_file.infolist()
                for file in all_files:
                    if not file.filename.endswith(".csv"):
                        logging.warning(f"Skipped zipped non-testcase file {file.filename}")
                        continue
                    logging.info(f"Reading zipped testcase file {file.filename}")
                    with zipped_file.open(file) as real_file:
                        testcase_manager = UniversalTestcaseFactory(io.TextIOWrapper(real_file,
                                                                                     encoding="UTF-8", newline=''))
                        sub_testcases = testcase_manager.get()
                        for group_hash in sub_testcases:
                            if group_hash in testcases:
                                original_length = len(testcases[group_hash])
                                testcases[group_hash] = testcases[group_hash].union(sub_testcases[group_hash])
                                if original_length + len(sub_testcases[group_hash]) != len(testcases[group_hash]):
                                    logging.warning("Possible zipped testcases hash collision detected! "
                                                    "Testcase integrity compromised")
                            else:
                                testcases[group_hash] = sub_testcases[group_hash]
        except zipfile.BadZipFile:
            logging.info("Input testcase file is not a valid zip file, switch to normal csv mode...")
        if not testcases:
            logging.info("Reading normal csv testcases...")
            with open(testcase_path, newline='') as file:
                testcase_manager = UniversalTestcaseFactory(file)
                testcases = testcase_manager.get()
        self.testcases = testcases
        self.flatten_testcases: Tuple[UniversalTestcaseStructure] = tuple_flatten(tuple(self.testcases.values()))
        if self.switches.preserve_original_csv:
            self.titles = testcase_manager.header
        else:
            self.titles = UniversalTestcaseFactory.get_all_visible_headers()
        if self.debug_mode and len(self.flatten_testcases) != 1:
            logging.error("Single testcase debugging mode cannot launch with more than one testcase!!!")
            raise RuntimeError("Single testcase debugging mode cannot launch with more than one testcase!!!")

    def _prepare_processes(self):
        # Create process for every usable device
        usable_devices = []
        for dev_logicid, dev_lock in enumerate(self.device_locks):
            if dev_lock is not None:
                usable_devices.append(dev_logicid)
            else:
                self.device_to_processes[dev_logicid] = ()
        if len(usable_devices) <= 0:
            raise RuntimeError("Available device count is zero, aborting.")
        if self.switches.process_per_device is None:
            self.switches.process_per_device = int(get_cpu_count() * 0.8) // len(usable_devices)
            self.switches.process_per_device = min(max(len(self.flatten_testcases) * 4 // len(usable_devices), 1),
                                                   self.switches.process_per_device,
                                                   32)
        logging.info(f"Process per device: {self.switches.process_per_device}")
        if self.switches.parallel_fatbin is None:
            if self.switches.process_per_device > 8:
                self.switches.parallel_fatbin = False
            else:
                self.switches.parallel_fatbin = True
        logging.info(f"Parallel fat-bin compilation: {self.switches.parallel_fatbin}")
        for logicid in usable_devices:
            self.device_to_processes[logicid] = tuple(SimpleCommandProcess(self.mp_context,
                                                                           name=f"D{logicid}P{i}")
                                                      for i in range(self.switches.process_per_device))
            for proc in self.device_to_processes[logicid]:
                self.process_to_device[proc] = logicid

    def _initialize_processes(self):
        while not all(proc.is_ready() for proc in self.process_to_device):
            self.__update_all_processes()

    def _prepare_result_titles(self):
        if self.switches.PMU:
            if self.switches.PMU_MODE == PMU_MODE.ADVANCED:
                self.dyn_pmu_titles = tuple("DYN_" + title for title in PMU_TITLES.ADVANCED_TITLES)
                self.stc_pmu_titles = tuple("STC_" + title for title in PMU_TITLES.ADVANCED_TITLES)
                self.cst_pmu_titles = tuple("CST_" + title for title in PMU_TITLES.ADVANCED_TITLES)
            else:
                self.dyn_pmu_titles = tuple("DYN_" + title for title in PMU_TITLES.DEFAULT_TITLES)
                self.stc_pmu_titles = tuple("STC_" + title for title in PMU_TITLES.DEFAULT_TITLES)
                self.cst_pmu_titles = tuple("CST_" + title for title in PMU_TITLES.DEFAULT_TITLES)
        self.result_titles = (ProfilingReturnStructure.__slots__[:-3]
                              + self.dyn_pmu_titles + self.stc_pmu_titles + self.cst_pmu_titles)

    def _prepare_tasks(self):
        task_pairs = list(enumerate(self.testcases.values()))
        random.shuffle(task_pairs)
        for group_id, testcases in task_pairs:
            for testcase in testcases:
                self.waiting_tasks.append(Task(group_id, testcase))
        self.total_tasks_count = len(self.waiting_tasks)

    def _update_processes(self):
        self.__update_all_processes()
        if time.time() - self.last_print_timestamp > self.print_time and self.switches.summary_print:
            title = (f"Version: {VERSION} Summary (Device Total: {self.switches.device_count}) "
                     f"Progress: {int(self.completed_task_count / self.total_tasks_count * 100)}% "
                     f"{self.completed_task_count} / {self.total_tasks_count} "
                     f"ET: {int(time.time() - self.start_timestamp)}s",)
            loop_count = self.switches.device_count // 2
            remain_count = self.switches.device_count % 2
            lines = [title]
            for loop in range(loop_count):
                lines.append((*self.__gen_info(loop * 2), *self.__gen_info(loop * 2 + 1)))
            if remain_count:
                lines.append((*self.__gen_info(loop_count * 2),))
            logging.info("\n" + table_print(lines))
            self.last_print_timestamp = time.time()
        # Check for completed process
        completed_process = []
        for proc in self.process_to_subtask:
            # Check if task completed
            if not proc.rpc_results.empty():
                completed_process.append(proc)
        for proc in completed_process:
            self.__handle_result(proc)
            del self.process_to_subtask[proc]
        # Check for subtasks and launch subtasks for them
        for dev_id in self.device_to_processes:
            for dev_proc in self.device_to_processes[dev_id]:
                if dev_proc not in self.process_to_subtask and \
                        dev_id in self.device_subtasks and self.device_subtasks[dev_id]:
                    subtask = self.device_subtasks[dev_id].pop()
                    self.process_to_subtask[dev_proc] = subtask
                    subtask.send_to_proc(dev_proc)

    def __update_all_processes(self):
        for dev_id in self.device_to_processes:
            for proc in self.device_to_processes[dev_id]:
                proc.update()

    # noinspection PyBroadException
    def __gen_info(self, device_id) -> tuple:
        if self.switches.mode.is_model():
            phyid = str(device_id)
            platform = self.switches.device_platform
            chip_ver = "MODEL"
            health = self.switches.mode.is_model()
            temperature = "???"
            aicore_freq = "????"
            aicore_rate = "????"
            aicore_util = "???"
        else:
            try:
                phyid = str(self.dsmi.get_physical_id_from_logical_id(device_id))
            except:
                phyid = str(device_id)
            try:
                dsmi_platform = self.dsmi.get_chip_info(device_id)
            except:
                platform = "ERR"
                chip_ver = "ERR"
            else:
                platform = dsmi_platform.get_complete_platform()
                chip_ver = dsmi_platform.get_ver()
            try:
                health = self.dsmi.get_device_health_state(device_id).name
            except:
                health = "ERR"
            try:
                temperature = str(self.dsmi.get_device_temperature(device_id))
            except:
                temperature = "???"
            try:
                aicore_freq = str(self.dsmi.get_device_frequency(device_id, 7))
            except:
                aicore_freq = "????"
            try:
                aicore_rate = str(self.dsmi.get_device_frequency(device_id, 9))
            except:
                aicore_rate = "????"
            try:
                aicore_util = str(self.dsmi.get_device_util(device_id, 2))
            except:
                aicore_util = "ERR"
        dev_info = f"{phyid.ljust(3)} " \
                   f"{platform.ljust(13)} " \
                   f"{chip_ver.ljust(2)} " \
                   f"{health}" \
                   f"\n{temperature.ljust(2)}C " \
                   f"{aicore_freq.ljust(5)}Mhz / " \
                   f"{aicore_rate.ljust(5)}Mhz " \
                   f"\nAICORE {aicore_util}%"
        proc_info = '\n'.join(f"{proc.get_pid()} "
                              f"{proc.name.ljust(20) if len(proc.name) < 20 else proc.name[:20]} "
                              f"{proc.data['stage'].ljust(20) if 'stage' in proc.data else 'UNKNOWN_STAGE'.ljust(20)} "
                              f"{proc.status.name.ljust(8)} "
                              f"{int(time.time() - proc.process_status_timestamp)}s"
                              for proc in self.device_to_processes[device_id])
        return dev_info, proc_info

    def _push_tasks(self):
        # Simple task pushing mechanism
        if self.waiting_tasks:
            for dev_id in self.device_to_processes:
                for dev_proc in self.device_to_processes[dev_id]:
                    if dev_proc not in self.process_to_subtask and self.waiting_tasks:
                        task = self.waiting_tasks.pop()
                        self.device_subtasks.setdefault(dev_id, []).append(SubTask(task, SubTaskType.COMPILE,
                                                                                   compilation_process,
                                                                                   (task.testcase_struct, task.group_id,
                                                                                    "dynamic")))
                        self.device_subtasks[dev_id].append(SubTask(task, SubTaskType.COMPILE,
                                                                    compilation_process,
                                                                    (task.testcase_struct, task.group_id, "static")))
                        self.device_subtasks[dev_id].append(SubTask(task, SubTaskType.COMPILE,
                                                                    compilation_process,
                                                                    (task.testcase_struct, task.group_id, "const")))
                        self.device_subtasks[dev_id].append(SubTask(task, SubTaskType.COMPILE,
                                                                    compilation_process,
                                                                    (task.testcase_struct, task.group_id, "binary")))
                        break

    def __handle_result(self, proc: SimpleCommandProcess):
        subtask = self.process_to_subtask[proc]
        result = proc.rpc_results.get()
        if subtask.type == SubTaskType.COMPILE:
            if isinstance(result, (DynamicCompilationResult, StaticCompilationResult,
                                   ConstCompilationResult, BinaryCompilationResult)):
                # Success
                self.__compile_on_completion(subtask, result)
            elif isinstance(result, SystemError):
                # Crashed
                self.__compile_on_crash(subtask, result)
            elif isinstance(result, RuntimeError) or result is None:
                # Failed
                self.__compile_on_failure(subtask, result)
            else:
                # Unknown
                raise RuntimeError(f"Unknown returning result {result}")
        elif subtask.type == SubTaskType.PROFILE:
            if isinstance(result, ProfilingReturnStructure):
                # Success
                self.__profile_on_completion(subtask, result)
            elif isinstance(result, SystemError):
                # Crashed
                self.__profile_on_crash(subtask, result)
            elif isinstance(result, RuntimeError):
                # Failed
                self.__profile_on_failure(subtask, result)
            else:
                # Unknown
                raise RuntimeError("Unknown returning result {result}")
            self.completed_task_count += 1
        else:
            raise RuntimeError(f"Invalid SubTask Type {subtask.type}")

    def __launch_profiling_process(self, task, dev_id: int):
        if self.switches.mode.is_model():
            related_testcase = copy.deepcopy(task.testcase_struct)
            related_testcase.model = True
            self.device_subtasks[dev_id].append(SubTask(task, SubTaskType.PROFILE,
                                                        profile_process,
                                                        (related_testcase, self.device_locks, dev_id)))
        else:
            task.testcase_struct.model = False
            self.device_subtasks[dev_id].append(SubTask(task, SubTaskType.PROFILE,
                                                        profile_process,
                                                        (task.testcase_struct, self.device_locks, dev_id)))

    def __compile_on_completion(self,
                                subtask: SubTask,
                                compile_actual_result):
        related_testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        task = subtask.task
        dev_id = self.process_to_device[subtask.process]
        if compile_actual_result is not None:
            compile_actual_result.apply(related_testcase)
        else:
            fail_reason = related_testcase.fail_reason if related_testcase.fail_reason \
                else related_testcase.dyn_fail_reason
            logging.warning(f"Compilation process of mode {subtask.params[2]} skipped for "
                            f"testcase {related_testcase.testcase_name} because of "
                            f"{fail_reason}")
            # Force ready
            related_testcase.ready_for_profile += 1
        if related_testcase.ready():
            self.__launch_profiling_process(task, dev_id)

    def __compile_on_crash(self,
                           subtask: SubTask,
                           _):
        related_testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        task = subtask.task
        mode = subtask.params[2]
        dev_id = self.process_to_device[subtask.process]
        logging.fatal(f"Compilation process crashed at stage {self.__get_process_stage_info(subtask.process)} "
                      f"for testcase {related_testcase.testcase_name} with pid {subtask.process.get_pid()}")
        crash_info = "Crashed at stage %s" % self.__get_process_stage_info(subtask.process)
        result = self.apply_errorinfo_to_testcase(crash_info, mode)
        result.apply(related_testcase)
        if related_testcase.ready():
            self.__launch_profiling_process(task, dev_id)
        subtask.process.resurrect()

    @staticmethod
    def apply_errorinfo_to_testcase(crash_info, mode):
        if mode == "dynamic":
            result = DynamicCompilationResult()
            result.tiling_result = DynamicOpTilingResult()
            result.tiling_result.all_set(crash_info)
        elif mode == "static":
            result = StaticCompilationResult()
        elif mode == "const":
            result = ConstCompilationResult()
        elif mode == "binary":
            result = BinaryCompilationResult()
            result.tiling_result = BinaryOpTilingResult()
            result.tiling_result.all_set(crash_info)
        else:
            logging.fatal("Unexpected compilation mode %s" % mode)
            raise RuntimeError("Unexpected compilation mode %s" % mode)
        result.all_set(crash_info)
        return result

    def __compile_on_failure(self,
                             subtask: SubTask,
                             compile_actual_result: Optional[RuntimeError]):
        # Get task exception message
        related_testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        task = subtask.task
        mode = subtask.params[2]
        dev_id = self.process_to_device[subtask.process]
        if isinstance(compile_actual_result, RuntimeError):
            exception_print: str = compile_actual_result.args[0]
            reason = "COMPILE_FAILURE"
            logging.error("Compilation process of mode %s failed for testcase %s, fail reason: \n%s"
                          % (mode, related_testcase.testcase_name, exception_print))
        else:
            reason = subtask.task.testcase_struct.fail_reason

        result = self.apply_errorinfo_to_testcase(reason, mode)
        result.apply(related_testcase)
        if related_testcase.ready():
            self.__launch_profiling_process(task, dev_id)
        subtask.process.resurrect()

    def __profile_on_crash(self,
                           subtask: SubTask,
                           _):
        testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        logging.fatal(f"Profile process crashed at stage {self.__get_process_stage_info(subtask.process)} "
                      f"for testcase {testcase.testcase_name}")
        basic_info = ("Crashed at profiling stage %s" % self.__get_process_stage_info(subtask.process),
                      *tuple("PROFILE_CRASH" for _ in range(len(self.result_titles) - 1)))
        if self.switches.preserve_original_csv:
            self.flush((*testcase.original_line,
                        "<-INPUT-|-RESULT->",
                        *basic_info))
        else:
            self.flush((*[getattr(testcase, title) for title in self.titles],
                        "<-INPUT-|-RESULT->",
                        *basic_info))
        subtask.process.resurrect()

    def __profile_on_failure(self,
                             subtask: SubTask,
                             compile_actual_result: RuntimeError):
        testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        exception_print = compile_actual_result.args[0]
        logging.error("Profiling process for testcase %s interrupted:\n%s"
                      % (testcase.testcase_name, exception_print))
        basic_info = (exception_print,  # Tiling time
                      *tuple("FAILURE" for _ in range(len(self.result_titles) - 1)))
        if self.switches.preserve_original_csv:
            self.flush((*testcase.original_line,
                        "<-INPUT-|-RESULT->",
                        *basic_info))
        else:
            self.flush((*[getattr(testcase, title) for title in self.titles],
                        "<-INPUT-|-RESULT->",
                        *basic_info))
        subtask.process.resurrect()

    def __profile_on_completion(self,
                                subtask: SubTask,
                                compile_actual_result: ProfilingReturnStructure):
        testcase: UniversalTestcaseStructure = subtask.task.testcase_struct
        prof_result: ProfilingReturnStructure = compile_actual_result
        if self.switches.preserve_original_csv:
            self.flush((*testcase.original_line,
                        "<-INPUT-|-RESULT->",
                        *prof_result.get()))
        else:
            self.flush((*[getattr(testcase, title) for title in self.titles],
                        "<-INPUT-|-RESULT->",
                        *prof_result.get()))
