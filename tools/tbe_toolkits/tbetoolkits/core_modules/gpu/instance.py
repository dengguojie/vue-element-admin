#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Main Sequence for dynamic shape profiling
"""
# Standard Packages
import csv
import time
import random
import logging
import multiprocessing
from typing import Set
from typing import Optional
from typing import Dict
from typing import List
from typing import Callable

# Third-Party Packages
from ..profiling import ProfilingInstance
from .profiling import profile_process

from ..tbe_multiprocessing.pool import SimpleCommandProcess

from ..testcase_manager import UniversalTestcaseFactory

from ...utilities import VERSION
from ...utilities import get_global_storage
from ...utilities import table_print
from ...utilities import tuple_flatten
from ..testcase_manager import UniversalTestcaseStructure


class Task:
    def __init__(self, testcase_struct: UniversalTestcaseStructure, func: Callable, params: tuple):
        self.testcase_struct = testcase_struct
        self.func = func
        self.params = params
        self.device_id = None
        self.process: Optional[SimpleCommandProcess] = None

    def send_to_proc(self, process: SimpleCommandProcess):
        process.send_action(self.func, self.params + (self.device_id,), {})
        self.process = process


class GPUProfilingInstance(ProfilingInstance):
    """
    GPU Profiling Interface
    """
    print_time = 5

    def __init__(self):
        super().__init__()
        # Switches Storage
        self.switches = get_global_storage()
        if self.switches.device_count == -1:
            self.switches.device_count = 8
        # gdb friendly mode
        self.debug_mode = self.switches.single_case_debugging
        # Testcases
        self.testcases: Optional[Dict[int, Set[UniversalTestcaseStructure]]] = None
        self.testcase_manager: Optional[UniversalTestcaseFactory] = None
        # Test Result CSV Storage
        self.titles: Optional[tuple] = None
        self.csv_writer = None
        self.result_titles = ()
        self.input_path: str = ""
        self.result_path: str = ""
        # Multiprocessing
        self.mp_context = multiprocessing.get_context("forkserver")
        self.device_to_process: Dict[int, SimpleCommandProcess] = {}
        self.process_to_device: Dict[SimpleCommandProcess, int] = {}
        # Initialized tasks without device
        self.waiting_tasks: List[Task, ...] = []
        self.total_tasks_count: int = 0
        self.completed_task_count: int = 0
        # Tasks executing
        self.process_to_task: Dict[SimpleCommandProcess, Task] = {}
        self.last_print_timestamp = time.time() - 20
        self.start_timestamp = time.time()

    def profile(self):
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
        if not self.result_path.endswith(".csv"):
            self.result_path += ".csv"
        logging.info("Preparing TBEToolkits GPU Profiling Infrastructure...")
        logging.info(f"Mode: {self.switches.mode.name}")
        logging.info(f"result_path: {self.result_path}")
        logging.info(f"Precision Comparison: NOT AVAILABLE")
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

    @staticmethod
    def __get_process_stage_info(proc: SimpleCommandProcess) -> str:
        return proc.data.get("stage")

    def _prepare_processes(self):
        # Create process for every usable device
        usable_devices = []
        for dev_logicid in range(self.switches.device_count):
            if dev_logicid not in self.switches.device_blacklist:
                usable_devices.append(dev_logicid)
        logging.info(f"Process per device: 1")
        for logicid in usable_devices:
            self.device_to_process[logicid] = SimpleCommandProcess(self.mp_context, name=f"GPU{logicid}")
            self.process_to_device[self.device_to_process[logicid]] = logicid

    def _initialize_processes(self):
        while not all(proc.is_ready() for proc in self.process_to_device):
            self.__update_all_processes()

    def _prepare_result_titles(self):
        self.result_titles = ("gpu_perf_us", "gpu_throughput_gbs")

    def _prepare_tasks(self):
        task_pairs = list(enumerate(self.testcases.values()))
        random.shuffle(task_pairs)
        for group_id, testcases in task_pairs:
            for testcase in testcases:
                self.waiting_tasks.append(Task(testcase, profile_process, (testcase,)))
        self.total_tasks_count = len(self.waiting_tasks)

    def _update_processes(self):
        self.__update_all_processes()
        if time.time() - self.last_print_timestamp > self.print_time:
            title = (f"Version: {VERSION} Summary (Device Total: {self.switches.device_count}) "
                     f"Progress: {int(self.completed_task_count / self.total_tasks_count * 100)}% "
                     f"{self.completed_task_count} / {self.total_tasks_count} "
                     f"ET: {time.time() - self.start_timestamp}s",)
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
        for proc in self.process_to_task:
            # Check if task completed
            if not proc.rpc_results.empty():
                completed_process.append(proc)
        for proc in completed_process:
            self.__handle_result(proc)
            del self.process_to_task[proc]
        # Check for tasks and launch tasks for them
        for dev_id in self.device_to_process:
            if self.device_to_process[dev_id] not in self.process_to_task and self.waiting_tasks:
                task = self.waiting_tasks.pop()
                task.device_id = dev_id
                self.process_to_task[self.device_to_process[dev_id]] = task
                task.send_to_proc(self.device_to_process[dev_id])

    def __update_all_processes(self):
        for dev_id in self.device_to_process:
            self.device_to_process[dev_id].update()

    def __gen_info(self, device_id) -> tuple:
        phyid = str(device_id)
        platform = "GPU"
        chip_ver = "??"
        health = "???"
        temperature = "???"
        aicore_freq = "????"
        aicore_rate = "????"
        aicore_util = "???"
        proc = self.device_to_process[device_id] if device_id in self.device_to_process else None
        dev_info = f"{phyid.ljust(3)} " \
                   f"{platform.ljust(13)} " \
                   f"{chip_ver.ljust(3)}" \
                   f"{health}" \
                   f"\n{temperature.ljust(2)}C " \
                   f"{aicore_freq.ljust(5)}Mhz / " \
                   f"{aicore_rate.ljust(5)}Mhz " \
                   f"\nAICORE {aicore_util}%"
        if proc is None:
            proc_info = ""
        else:
            proc_info = (f"{proc.get_pid()} "
                         f"{proc.name.ljust(20) if len(proc.name) < 20 else proc.name[:20]} "
                         f"{proc.data['stage'].ljust(20) if 'stage' in proc.data else 'UNKNOWN_STAGE'.ljust(20)} "
                         f"{proc.status.name.ljust(8)} "
                         f"{int(time.time() - proc.process_status_timestamp)}s")
        return dev_info, proc_info

    def __handle_result(self, proc: SimpleCommandProcess):
        task = self.process_to_task[proc]
        result = proc.rpc_results.get()
        if isinstance(result, tuple):
            self.__profile_on_completion(task, result)
        elif isinstance(result, SystemError):
            # Crashed
            self.__profile_on_crash(task, result)
        elif isinstance(result, RuntimeError):
            # Failed
            self.__profile_on_failure(task, result)
        else:
            # Unknown
            raise RuntimeError("Unknown returning result {result}")
        self.completed_task_count += 1


    def __profile_on_crash(self,
                           task: Task,
                           _):
        testcase: UniversalTestcaseStructure = task.testcase_struct
        crashed_stage = self.__get_process_stage_info(task.process)
        logging.fatal(f"Profile process crashed at stage {crashed_stage} "
                      f"for testcase {testcase.testcase_name}")
        basic_info = (f"Crashed at profiling stage {crashed_stage}",
                      *tuple("PROFILE_CRASH" for _ in range(len(self.result_titles) - 1)))
        self.flush((*[getattr(testcase, title) for title in self.titles],
                    "<-INPUT-|-RESULT->",
                    *basic_info))
        task.process.resurrect()

    def __profile_on_failure(self,
                             task: Task,
                             compile_actual_result: RuntimeError):
        testcase: UniversalTestcaseStructure = task.testcase_struct
        exception_print = compile_actual_result.args[0]
        logging.error("Profiling process for testcase %s interrupted:\n%s"
                      % (testcase.testcase_name, exception_print))
        basic_info = tuple("FAILURE" for _ in range(len(self.result_titles)))
        self.flush((*[getattr(testcase, title) for title in self.titles],
                    "<-INPUT-|-RESULT->",
                    *basic_info))

    def __profile_on_completion(self,
                                task: Task,
                                compile_actual_result: tuple):
        testcase: UniversalTestcaseStructure = task.testcase_struct
        self.flush((*[getattr(testcase, title) for title in self.titles],
                    "<-INPUT-|-RESULT->",
                    *compile_actual_result))
