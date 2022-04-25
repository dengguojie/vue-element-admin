#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Custom Pool Classes
"""
# Standard Packages
import os
import sys
import time
import logging
import traceback
import subprocess
import multiprocessing as mp
from enum import Enum
from enum import auto
from queue import SimpleQueue
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import NoReturn
from typing import Callable

# Third-Party Packages
import psutil
from ...utilities import set_global_storage
from ...utilities import get_global_storage
from ..tbe_logging import default_logging_config


class PROCESS_STATUS_CODE(Enum):
    CREATED = 0
    LAUNCHED = 114
    DEBUG_ATTACH = 514
    READY = 1919
    RUNNING = 810
    WAITING = 1551
    DEAD = 1145141919810


class PROCESS_RPC(Enum):
    EXECUTE_FUNCTION = auto()
    FUNCTION_RETURN = auto()
    ACQUIRE_SEMAPHORE = auto()
    SET_SEMAPHORE = auto()
    GET_SEMAPHORE = auto()
    RELEASE_SEMAPHORE = auto()
    GET_ACQUIRED_SEMAPHORES = auto()
    STORE_DATA = auto()
    CHANGE_NAME = auto()
    SUICIDE = auto()
    SEMAPHORE_DEAD_SEQUENCE = auto()
    GET_LOCK = auto()
    RELEASE_LOCK = auto()


process_context = None


def get_process_context() -> "Optional[ProcessContext]":
    return process_context


def intermediate_func(pipe: "mp.connection.Connection", debug_mode: bool, global_storage) -> NoReturn:
    pipe.send(PROCESS_STATUS_CODE.LAUNCHED)  # 0x0114 -> Launched
    set_global_storage(global_storage)
    default_logging_config(file_handler=get_global_storage().logging_to_file)
    global process_context
    process_context = ProcessContext(pipe)
    while True:
        pipe.send(PROCESS_STATUS_CODE.READY)  # 0x1919 -> Ready for Command
        # noinspection PyBroadException
        try:
            message = pipe.recv()
        except:
            # Silently suicide
            sys.exit(-1)
        if isinstance(message, tuple):
            # Prepare for RPC
            rpc_command = message[0]
            if rpc_command == PROCESS_RPC.EXECUTE_FUNCTION:
                if debug_mode:
                    pipe.send(PROCESS_STATUS_CODE.DEBUG_ATTACH)  # 0x0514 -> Waiting for Attach
                    logging.info("Process waiting for attach")
                    time.sleep(15)  # Waiting for attach
                    logging.info("Wait complete")
                rpc_args = message[1]
                pipe.send(PROCESS_STATUS_CODE.RUNNING)
                # noinspection PyBroadException
                try:
                    func: Callable = rpc_args[0]
                    func_args: tuple = rpc_args[1]
                    func_kwargs: dict = rpc_args[2]
                    function_return = func(*func_args, **func_kwargs)
                    pipe.send((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                    del func, func_args, func_kwargs
                except:
                    process_context.send_data("stage", "OnExceptionReport")
                    function_return = RuntimeError(traceback.format_exc())
                    pipe.send((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                    get_process_context().semaphore_dead_process(function_return)
                    pipe.send(PROCESS_STATUS_CODE.DEAD)
                    pipe.close()
                    return
            elif rpc_command == PROCESS_RPC.SUICIDE:
                sys.exit(-1)
            else:
                logging.warning(f"SimpleCommandProcess command pipe received invalid rpc call, ignored: {message}")
        else:
            logging.warning(f"SimpleCommandProcess command pipe received invalid command, ignored: {message}")


def get_cpu_count() -> int:
    return len(os.sched_getaffinity(0))


class ProcessContext:
    def __init__(self, pipe):
        self.pipe: "mp.connection.Connection" = pipe
        self.storage = {}

    def acquire_semaphore(self, name) -> bool:
        self.pipe.send((PROCESS_RPC.ACQUIRE_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def set_semaphore(self, name, value) -> NoReturn:
        self.pipe.send((PROCESS_RPC.SET_SEMAPHORE, (name, value)))

    def get_semaphore(self, name) -> NoReturn:
        self.pipe.send((PROCESS_RPC.GET_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def get_acquired_semaphore(self) -> tuple:
        self.pipe.send((PROCESS_RPC.GET_ACQUIRED_SEMAPHORES, ()))
        return self.pipe.recv()

    def semaphore_dead_process(self, value):
        self.pipe.send((PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE, (value,)))

    def send_data(self, name, value) -> NoReturn:
        self.pipe.send((PROCESS_RPC.STORE_DATA, (name, value)))

    def change_name(self, name: str) -> NoReturn:
        self.pipe.send((PROCESS_RPC.CHANGE_NAME, (name,)))

    def get_lock(self, lock) -> NoReturn:
        self.pipe.send((PROCESS_RPC.GET_LOCK, (lock,)))

    def release_lock(self, lock) -> NoReturn:
        self.pipe.send((PROCESS_RPC.RELEASE_LOCK, (lock,)))


class SimpleCommandProcess:
    semaphore_to_holder: Dict[Any, "SimpleCommandProcess"] = {}
    holder_to_semaphores: Dict["SimpleCommandProcess", List[Any]] = {}
    semaphores: Dict[Any, Any] = {}
    all_processes: List["SimpleCommandProcess"] = []

    def __init__(self, context=mp, name="TBESimpleCommandProcess", daemon=None, debug_mode=False):
        self.original_input_params = (context, name, daemon, debug_mode)
        self.status: PROCESS_STATUS_CODE = PROCESS_STATUS_CODE.CREATED
        self.debug_mode = debug_mode
        self.rpc_queue: SimpleQueue = SimpleQueue()
        self.rpc_results: SimpleQueue = SimpleQueue()
        self.locks: list = []
        self.process_status_timestamp = time.time()
        self.data: Dict[str, Any] = {}
        self.parent_pipe, self.child_pipe = context.Pipe()
        self.name = name
        self.parent = context.Process(target=intermediate_func, name=name, args=(self.child_pipe, debug_mode,
                                                                                 get_global_storage()),
                                      daemon=daemon)
        self.all_processes.append(self)
        self.parent.start()
        logging.debug(f"Process created with name {self.parent.name}")

    @staticmethod
    def _handle_locks():
        lock = None
        if isinstance(lock, mp.synchronize.Semaphore):
            raise RuntimeError("Subprocess dead while holding Semaphore")
        elif isinstance(lock, mp.synchronize.Lock):
            # noinspection PyBroadException
            try:
                lock.release()
            except:
                logging.exception("Lock releasing failure:")
        elif isinstance(lock, mp.synchronize.RLock):
            raise RuntimeError("Subprocess dead while holding RLock")

    def _update(self):
        while self.parent_pipe.poll():
            message = self.parent_pipe.recv()
            if isinstance(message, PROCESS_STATUS_CODE):
                self.status = message
                self.process_status_timestamp = time.time()
                if message == PROCESS_STATUS_CODE.DEAD:
                    return
            elif isinstance(message, tuple):
                rpc_command = message[0]
                rpc_args = message[1]
                self._rpc_call(rpc_command, rpc_args)
            else:
                raise RuntimeError(f"SimpleCommandProcess received invalid command: {message}")
        if self.get_exitcode() is not None:
            raise EOFError()

    def _parent_send_rpc(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        self.parent_pipe.send((rpc_command, rpc_args))

    def _clear_data(self):
        self.data = {}

    def _rpc_call(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        if rpc_command == PROCESS_RPC.EXECUTE_FUNCTION:
            func = rpc_args[0]
            func_args = rpc_args[1]
            func_kwargs = rpc_args[2]
            function_return = func(*func_args, **func_kwargs)
            self._parent_send_rpc(PROCESS_RPC.FUNCTION_RETURN, (function_return,))
        elif rpc_command == PROCESS_RPC.FUNCTION_RETURN:
            function_return = rpc_args[0]
            self.rpc_results.put(function_return)
            self._clear_data()
            self.name = self.original_input_params[1]
            self.parent.name = self.name
        elif rpc_command == PROCESS_RPC.STORE_DATA:
            name: str = rpc_args[0]
            value = rpc_args[1]
            self.data[name] = value
        elif rpc_command == PROCESS_RPC.CHANGE_NAME:
            name: str = rpc_args[0]
            self.name = name
            self.parent.name = name
        elif rpc_command == PROCESS_RPC.ACQUIRE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphore_to_holder:
                self.parent_pipe.send(False)
            else:
                self.semaphore_to_holder[name] = self
                self.holder_to_semaphores.setdefault(self, []).append(name)
                self.semaphores[name] = None
                self.parent_pipe.send(True)
        elif rpc_command == PROCESS_RPC.SET_SEMAPHORE:
            name = rpc_args[0]
            value = rpc_args[1]
            if self.semaphore_to_holder[name] == self:
                self.semaphores[name] = value
            else:
                logging.warning(f"{self.name} trying to access semaphore of another process: {name}")
        elif rpc_command == PROCESS_RPC.GET_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphores:
                value = self.semaphores[name]
            else:
                value = None
            self.parent_pipe.send(value)
        elif rpc_command == PROCESS_RPC.RELEASE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphores and self.semaphore_to_holder[name] == self:
                del self.semaphores[name]
                del self.semaphore_to_holder[name]
                self.holder_to_semaphores[self].remove(name)
            else:
                logging.warning(f"{self.name} trying to release invalid semaphore: {name}")
        elif rpc_command == PROCESS_RPC.GET_ACQUIRED_SEMAPHORES:
            if self in self.holder_to_semaphores:
                self.parent_pipe.send(self.holder_to_semaphores[self])
            else:
                self.parent_pipe.send(())
        elif rpc_command == PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE:
            value = rpc_args[0]
            self._semaphore_dead_sequence(value)
        elif rpc_command == PROCESS_RPC.GET_LOCK:
            lock = rpc_args[0]
            logging.debug(f"Get lock {lock}")
            self.locks.append(lock)
        elif rpc_command == PROCESS_RPC.RELEASE_LOCK:
            lock = rpc_args[0]
            logging.debug(f"Release lock {lock}")
            found = False
            for _lock in self.locks:
                # noinspection PyProtectedMember
                if _lock._id == lock._id:
                    found = True
                    self.locks.remove(_lock)
                    break
            if not found:
                logging.warning(f"{self.name} trying to release unknown lock: {lock}")
        else:
            raise NotImplementedError(f"SimpleCommandProcess Master received invalid rpc call: "
                                      f"{rpc_command, rpc_args}")

    def _semaphore_dead_sequence(self, value):
        if self in self.holder_to_semaphores:
            for sem in self.holder_to_semaphores[self]:
                if self.semaphores[sem] is None:
                    self.semaphores[sem] = value

    def get_pid(self):
        try:
            return self.parent.pid
        except ValueError:
            return None

    def get_exitcode(self):
        return self.parent.exitcode

    def get_memory_usage_percent(self):
        return psutil.Process(self.get_pid()).memory_percent()

    def resurrect(self):
        self.__init__(*self.original_input_params)

    def close(self):
        if not self.status == PROCESS_STATUS_CODE.DEAD:
            # noinspection PyBroadException
            try:
                self._parent_send_rpc(PROCESS_RPC.SUICIDE, ())
            except:
                pass
            else:
                self.parent.terminate()
                self.parent.join()
            finally:
                if self.get_exitcode() is not None:
                    self.kill()
                    self.parent.join()
                self.parent.close()
                self.status = PROCESS_STATUS_CODE.DEAD

    def kill(self):
        self.parent.kill()

    def update(self) -> NoReturn:
        if self.status == PROCESS_STATUS_CODE.DEAD:
            return
        try:
            self._update()
        except EOFError:
            # Check for exitcode
            if self.get_exitcode() is not None:
                logging.warning(f"Process {self.name} exited unexpectedly with code {self.get_exitcode()}")
                exception = SystemError(f"Process {self.name} exited unexpectedly with code {self.get_exitcode()}")
                self.rpc_results.put(exception)
                self.close()
            else:
                logging.warning(f"Process {self.name} lost connection with the parent process")
                exception = SystemError(f"Process {self.name} lost connection with the parent process")
                self.rpc_results.put(exception)
                self.kill()
                self.close()
            self._semaphore_dead_sequence(exception)
            for lock in self.locks:
                # noinspection PyBroadException
                try:
                    lock.release()
                except:
                    logging.exception("Release lock failed:")
        except:
            self.close()
            raise
        if self.status == PROCESS_STATUS_CODE.READY and not self.rpc_queue.empty():
            func_call = self.rpc_queue.get()
            self._parent_send_rpc(*func_call)
            self.status = PROCESS_STATUS_CODE.WAITING
            if self.debug_mode:
                choice = input(f"You are now trying to attach to {func_call[1]}, input (y or Y) if you want to:")
                if choice.lower() == "y":
                    subprocess.run(["gdb", "attach", str(self.get_pid())], shell=False)

    def send_action(self, target: Callable, args: tuple, kwargs: dict):
        self.rpc_queue.put((PROCESS_RPC.EXECUTE_FUNCTION, (target, args, kwargs)))

    def is_ready(self):
        if self.status == PROCESS_STATUS_CODE.READY:
            return True
        return False
