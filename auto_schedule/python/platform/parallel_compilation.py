# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
support parallel compilation
"""
import multiprocessing as mp
import queue
import time
import os
import stat
import importlib
import signal
import datetime
import traceback
import shutil
import zlib
import pickle
import sys
import subprocess
import logging
import json
import threading
from pathlib import Path
from configparser import ConfigParser

import te.platform.log_util as te_log
import te.platform.fusion_manager as fusion_manager
import te.platform.fusion_util as fusion_util
import te.platform.cce_policy as cce_policy

FILE_MODE_440 = stat.S_IRUSR | stat.S_IRGRP
FLAG = os.O_WRONLY | os.O_CREAT
MAXINT32 = 2**32


def init_logger():
    """
    init logger module
    """
    logging.raiseExceptions = False
    logmap = {
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }
    loglevel = os.getenv('TE_LOGLEVEL', 'WARNING')
    loglevel = logmap[loglevel] if loglevel in logmap else logging.WARNING

    newlogger = logging.getLogger('PCOMPILE')
    newlogger.propagate = False
    newlogger.setLevel(loglevel)
    log_ch = logging.StreamHandler()
    log_fmt = logging.Formatter('%(asctime)s %(process)d %(name)s %(message)s')
    log_ch.setFormatter(log_fmt)
    newlogger.addHandler(log_ch)
    return newlogger


logger = init_logger()          # pylint: disable=invalid-name


# 'pylint: disable=too-few-public-methods
class Counter:
    """
    Atomic counter
    """
    counter = MAXINT32
    locker = threading.Lock()

    @staticmethod
    def next():
        """
        get next counter
        :return: next counter
        """
        with Counter.locker:
            Counter.counter += 1
        return Counter.counter


def mygetattr(obj, name):
    """
    get object attr recursively
    :param obj: python object
    :param name: attr name
    :return: attr
    """
    if not name:
        return obj
    name_list = name.split(".")
    while name_list:
        obj = getattr(obj, name_list[0])
        name_list = name_list[1:]
    return obj


def mysetattr(obj, name, value):
    """
    get object attr recursively
    :param obj:  python object
    :param name: attr name
    :param value: attr value
    :return: None
    """
    name_list = name.split(".")
    target_obj = mygetattr(obj, ".".join(name_list[:-1]))
    target_attr = name_list[-1]
    setattr(target_obj, target_attr, value)


def excepthook_silent(etype, value, tback):  # pylint: disable=unused-argument
    """
    excepthook to print nothing
    """


def worker_sigint_handler(signum, frame):  # pylint: disable=unused-argument
    """
    worker process just quit when Ctrl-C pressed
    """
    # logging module uses reentrant threading.Rlock,
    # can be safely used in signal handler
    logger.warning('Ctrl-C pressed or session exit or cur python terminate,'
                   ' worker process quiting...')
    del_tmp_files_by_pid(OpCompiler.master_pid)
    os._exit(1)  # pylint: disable=protected-access


def worker_sigint_handler_no_print(signum, frame):  # pylint: disable=unused-argument
    """
    worker process just quit when Ctrl-C pressed
    """
    # logging module uses reentrant threading.Rlock,
    # can be safely used in signal handler
    del_tmp_files_by_pid(OpCompiler.master_pid)
    os._exit(1)  # pylint: disable=protected-access


def exec_compilation_task(worker_env, task_env):
    """
    compilation task worker entry
    :param socinfo: soc_version, core_type, core_num, l1size
    :param task_env: tuple of task queue, pipe, etc...
    :return: None
    """
    # just quit, avoid KeyboardInterrupt traceback printing mess
    sig_list = [signal.SIGINT, signal.SIGHUP]
    for sig in sig_list:
        signal.signal(sig, worker_sigint_handler)
    signal.signal(signal.SIGTERM, worker_sigint_handler_no_print)

    cce = importlib.import_module("te.platform.cce_conf")

    socinfo, dispatcher, autotune_dispatcher,\
        pid, worker_idx, slog_level, slog_event = worker_env
    try:
        from te.utils.AscendLog import AscendLog
        slog = AscendLog()
        if slog_level is not None:
            slog.set_level(-1, slog_level, slog_event)
    except Exception:       # 'pylint: disable=broad-except
        pass

    logger.info("socinfo:%s", socinfo)
    cce.te_set_version(*socinfo)
    OpCompiler.task_dispatcher = dispatcher
    OpCompiler.autotune_task_dispatcher = autotune_dispatcher
    OpCompiler.master_pid = pid
    OpCompiler.autotune_worker_idx = worker_idx

    importlib.import_module("te.platform.fusion_manager")
    importlib.import_module("te.platform.fusion_util")
    worker = TaskWorker(task_env)

    logger.info("Default compiler:%s", OpCompiler.compiler)
    logger.info("Autotune compiler:%s", OpCompiler.autotune_compiler)

    worker.loop()


def get_multi_process_count(tune_mode):
    """
    get compilation worker number from ini conf file
    :return: compilation worker number
    """
    try:
        if 'TE_PARALLEL_COMPILER' in os.environ:
            count = int(os.getenv('TE_PARALLEL_COMPILER'))
            logger.info("TE_PARALLEL_COMPILER=%s", count)
        else:
            home_path = os.getenv('HOME')
            config_file_path = os.path.join(home_path, ".tbe_build.ini")
            config_file = ConfigParser()
            config_file.read(config_file_path)
            count = config_file.getint('compilation', 'max_parallel_jobs')
        count = max(0, min(count, len(os.sched_getaffinity(0))))
        # always enable async when RL Tune
        if "RL" in tune_mode and count == 0:
            count = 1
        return count
    except Exception:       # 'pylint: disable=broad-except
        return 8


def set_main_info():
    """
    set __file__ and name of main to None
    :return: (orignal main module name, path)
    """
    main_module = sys.modules['__main__']
    main_mod_name = getattr(main_module.__spec__, "name", None)
    main_path = getattr(main_module, '__file__', None)
    if main_mod_name is not None:
        setattr(main_module.__spec__, "name", None)

    if main_path is not None:
        setattr(main_module, '__file__', None)
    return (main_mod_name, main_path)


def restore_main_info(name, path):
    """
    restor main module name and path
    """
    main_module = sys.modules['__main__']
    if name is not None:
        setattr(main_module.__spec__, "name", name)
    if path is not None:
        setattr(main_module, '__file__', path)


def guess_pyexe_path(mp_ctx):
    """
    search for a suitable python exe, should be called before any multiprocessing calls
    :param mp_ctx: multiprocessing module
    """
    pylibver = sys.version_info
    pyver = subprocess.run([sys.executable, '-V'], stderr=subprocess.DEVNULL,
                           check=False,
                           stdout=subprocess.PIPE).stdout.decode().split()[1].split('.')
    if pyver[0] == str(pylibver.major) and pyver[1] == str(pylibver.minor):
        return

    targetpy = "python" + str(pylibver.major) + "." + str(pylibver.minor)
    binpath = [os.path.join(path, targetpy) for path in
               os.environ['PATH'].split(os.pathsep) + ['/usr/bin', '/usr/local/bin']]

    for path in binpath:
        if os.path.isfile(path):
            mp_ctx.set_executable(path)
            logger.info("guessed python path:%s", path)
            return


class OpCompiler:
    """
    OpCompiler
    """
    compiler = None
    autotune_compiler = None
    worker_checker_count = 0
    master_pid = 0
    task_dispatcher = None
    autotune_task_dispatcher = None
    autotune_worker_idx = -1
    atc_time_stamp = None

    def __init__(self, embedding, worker_num, worker_env, autotune=False,
                 time_stamp=None, slog_level=None, slog_event=1):
        """
        init
        :param task_env:
        :param worker_list:
        """
        self.task_dispatcher = None
        # '{graphid: {taskid: desc}}
        self._task_running = {}

        # '{graphid: {taskid: result}}
        self._task_finished = {}
        self._worker_num = worker_num
        self._worker_env = worker_env
        self._embedding = embedding
        self.finished_task_queue = None
        self.live_checker = None
        self.termination_event = None
        self.extra_res_queue = {}
        self._worker_list = []
        self.data_queue = []
        self.task_queue = []
        self._autotune = autotune
        self._time_stamp = time_stamp
        self._slog_level = slog_level
        self._slog_event = slog_event
        if autotune:
            OpCompiler.autotune_compiler = self
        else:
            OpCompiler.compiler = self
        OpCompiler.master_pid = os.getpid()

    def init(self):
        """
        init task queue, data queue and result queue
        """
        if self.task_dispatcher is not None:
            return

        ctx = mp.get_context("forkserver")
        self.task_queue = ctx.Queue()
        self.task_queue.cancel_join_thread()
        if self._autotune:
            # Each autotune sub process has it's own finished task queue
            self.finished_task_queue = [ctx.Queue() for
                                        _ in range(0, self._worker_num)]
            for tqueue in self.finished_task_queue:
                tqueue.cancel_join_thread()
        else:
            self.finished_task_queue = ctx.Queue()
            self.finished_task_queue.cancel_join_thread()
        self.termination_event = ctx.Event()
        self.live_checker = ctx.Pipe()
        self.data_queue = [ctx.Queue() for
                           worker in range(0, self._worker_num)]
        for dqueue in self.data_queue:
            dqueue.cancel_join_thread()
        self.task_dispatcher = \
            TaskDispatcher((self.task_queue, self.finished_task_queue,
                            self.data_queue))

    def start(self):
        """
        start worker compiler process
        """
        if self._worker_list:
            return self._worker_num, self.finished_task_queue, \
                self.live_checker, self.termination_event

        if self._embedding:
            guess_pyexe_path(mp)

        # multiprocessing will access sys.argv, if sys.argv not exist,
        # exception raised and can not be caught here
        if not hasattr(sys, "argv"):
            sys.argv = ['']

        # Child process of py multiprocessing will import all modules imported
        # by parent, which is unnecessary and problematic, here is a hack to
        # bypass it.
        main_mod_name, main_path = set_main_info()
        autotune_dispatcher = None
        if OpCompiler.autotune_compiler is not None:
            autotune_dispatcher = OpCompiler.autotune_compiler.task_dispatcher
        for idx in range(0, self._worker_num):
            data_queue = self.data_queue[idx]
            ctx = mp.get_context("forkserver")
            worker = \
                ctx.Process(target=exec_compilation_task,
                            args=(
                                (self._worker_env,
                                 OpCompiler.compiler.task_dispatcher,
                                 autotune_dispatcher,
                                 OpCompiler.master_pid, idx, self._slog_level,
                                 self._slog_event),
                                (self.task_queue, self.finished_task_queue,
                                 data_queue, self.termination_event,
                                 self.live_checker[0], self._time_stamp)
                            ),
                            daemon=True)
            worker.start()
            self._worker_list.append(worker)
        restore_main_info(main_mod_name, main_path)

        return self._worker_num, self.finished_task_queue, self.live_checker, \
            self.termination_event

    def destory(self):
        """
        deinit multi compilation process
        :return: None
        """
        dispatcher = self.task_dispatcher
        if dispatcher is None:
            return
        self.termination_event.set()

        try:
            time.sleep(0.2)
            for worker in self._worker_list:
                if worker.is_alive():
                    worker.terminate()
        except Exception:       # 'pylint: disable=broad-except
            # Sub processes may already quit when being killed
            pass
        self.task_dispatcher = None
        self._task_running = {}
        OpCompiler.compiler = None
        OpCompiler.autotune_compiler = None
        OpCompiler.master_pid = 0

    def is_worker_alive(self):
        """
        check wether all worker processes are alive
        :return:
        """
        if self.task_dispatcher is None:
            return False
        all_alive = True
        for worker in self._worker_list:
            if not worker.is_alive():
                logger.warning("worker process %s died. exitcode %s",
                               worker.pid, worker.exitcode)
                all_alive = False
        return all_alive

    def check_worker_status(self):
        """
        check if worker process are alive, if not, set all running task as fail
        """
        # if any worker process dead, all task will be markded as failed
        OpCompiler.worker_checker_count += 1
        if OpCompiler.worker_checker_count % 3000 != 0:
            return

        worker_alive = self.is_worker_alive()
        autotune_worker_alive = True if OpCompiler.autotune_compiler is None \
            else OpCompiler.autotune_compiler.is_worker_alive()
        if not worker_alive or not autotune_worker_alive:
            for gid, tasks in list(self._task_running.items()):
                for tid, task_desc in list(tasks.items()):
                    errmsg = "compiler process died"
                    task_res = gen_task_res(0, gid, tid, 1,
                                            'FatalError', errmsg,
                                            err_args=task_desc)
                    self.save_finished_task(gid, tid, task_res)
            self._task_running.clear()

    def get_finished_task(self, graphid=None, taskids=None):
        """
        return finished compilation task
        :return:
        """
        if self.task_dispatcher is None:
            return []
        try:
            while True:
                task_res = self.task_dispatcher.get_result(False)
                gid = task_res['graph_id']
                tid = task_res['task_id']
                self.save_finished_task(gid, tid, task_res)
        except queue.Empty:
            pass

        self.check_worker_status()

        res = []
        if graphid is not None:
            task_res = self._task_finished.get(graphid)
            if task_res is None:
                return res

            if taskids is None:
                res = list(task_res.values())
                del self._task_finished[graphid]
            else:
                res = [task_res.pop(tid, None) for tid in taskids]
                if task_res == {}:
                    del self._task_finished[graphid]
        else:
            for gid, tasks in self._task_finished.items():
                res.extend(list(tasks.values()))
            self._task_finished.clear()

        return res

    def update_running_task(self, task):
        """
        update task to _task_running
        :param task:
        """
        if self.task_dispatcher is None:
            return
        runnings = self._task_running.setdefault(task.graph_id, {})
        running = runnings.get(task.task_id)
        if running is not None:
            logger.warning("task already exist, dispatch failed. %d:%d",
                           task.graph_id, task.task_id)
            return
        runnings[task.task_id] = task.desc()

    def dispatch_task(self, task):
        """
        dispatch task to workers
        :param task:
        """
        if self.task_dispatcher is None:
            return
        self.task_dispatcher.dispatch(task)
        self.update_running_task(task)

    def sync_data(self, data):
        """
        sync data to all workers
        :param data:
        """
        if self.task_dispatcher is None:
            return
        self.task_dispatcher.sync_data(data)

    def clear_running_task(self, gid, tid):
        """
        clear running task
        :param gid: task graphid
        :param tid: task taskid
        """
        if self.task_dispatcher is None:
            return True
        tasks_in_gid = self._task_running.get(gid)
        if tasks_in_gid is None:
            logger.info("task finished, but graphid not found. %d:%d",
                        gid, tid)
            return False

        running = tasks_in_gid.get(tid)
        if running is None:
            logger.info("task finished, but taskid not found. %d:%d",
                        gid, tid)
            return False

        del tasks_in_gid[tid]
        return True

    def save_finished_task(self, gid, tid, res):
        """
        save finished task
        :param gid: task graphid
        :param tid: task taskid
        :param res: task result
        """
        if self.task_dispatcher is None:
            return
        self.clear_running_task(gid, tid)

        finished_task_in_gid = self._task_finished.setdefault(gid, {})
        finished_task_in_gid[tid] = res


# 'pylint: disable=too-few-public-methods
class DeferredOpRes:
    """
    DeferredOpRes
    """
    _task_finished = {}

    def __init__(self, gid, tid, res=None, tag=None, from_worker=-1):
        """
        init DeferredOpRes
        :param gid
        :param tid
        """
        self._gid = gid
        self._tid = tid
        self._res = res
        self._tag = tag
        self._from_worker = from_worker

    def get(self):
        """
        get Op compilation result
        :return: None if still runing, others if finished
        """
        if self._res is not None:
            return self._res

        if self._tag == "autotune_compile_op":
            try:
                while True:
                    autotune_dispatcher = OpCompiler.autotune_task_dispatcher
                    task_res = autotune_dispatcher\
                        .get_result(False, self._from_worker)
                    gid = task_res['graph_id']
                    tid = task_res['task_id']
                    DeferredOpRes._task_finished[(gid, tid)] = task_res
            except queue.Empty:
                pass

            res = DeferredOpRes._task_finished.get((self._gid, self._tid),
                                                   None)
            if res:
                res = [res]
            else:
                res = []
        else:
            compiler = OpCompiler.compiler
            res = compiler.get_finished_task(self._gid, [self._tid])

        if len(res) == 0:
            return None
        res = res[0]
        if res is not None:
            self._res = res

        return self._res


def init_multi_process_env(embedding, socinfo, tune_mode,
                           slog_level=None, slog_event=1,
                           pid_timestamp=""):
    """
    init multi compilation process
    :param embedding: if is embedding python
    :param socinfo:
    :param l2mode:
    :return: compilation worker number
    """
    logger.info("pid_timestamp: %s", pid_timestamp)
    atc_time_stamp = pid_timestamp
    OpCompiler.atc_time_stamp = atc_time_stamp
    process_count = get_multi_process_count(tune_mode)
    if process_count <= 0:
        return 0, None, None, None

    compiler = OpCompiler(embedding, process_count, socinfo,
                          slog_level=slog_level,
                          slog_event=slog_event)
    if 'GA' in tune_mode:
        autotune_compiler = OpCompiler(embedding, process_count, socinfo, True,
                                       atc_time_stamp,
                                       slog_level=slog_level,
                                       slog_event=slog_event)
    compiler.init()
    if 'GA' in tune_mode:
        autotune_compiler.init()

    res = compiler.start()
    if 'GA' in tune_mode:
        autotune_compiler.start()
    return res


def del_tmp_files_by_pid(pid):
    """
    deleate tmp files by atc pid
    :param: atc pid
    :return: None
    """
    cur_path = os.getcwd()
    tmp_list = []
    file_lock_path = os.path.join(cur_path, "file.lock")
    tmp_list.append(file_lock_path)
    for path_item in Path(cur_path).glob('*pid{}*'.format(pid)):
        list_item = os.fspath(path_item)
        if "tune_result" not in list_item:
            tmp_list.append(list_item)
    try:
        for item in tmp_list:
            if os.path.isfile(item):
                os.remove(item)
            if os.path.isdir(item):
                shutil.rmtree(item)
    except OSError:
        pass


def del_tmp_files(atc_time_stamp):
    """
    deleate tmp files in the atc process
    :param: atc_time_stamp, the atc pid and time stamp str
    :return: None
    """
    cur_path = os.getcwd()
    tune_show_dir_name = "tune_show_{}".format(atc_time_stamp)
    lock_file_name = "file.lock"
    tune_show_dir = os.path.join(cur_path, tune_show_dir_name)
    lock_file = os.path.join(cur_path, lock_file_name)
    if os.path.isfile(lock_file):
        os.remove(lock_file)
    if os.path.isdir(tune_show_dir):
        shutil.rmtree(tune_show_dir)


def deinit_multi_process_env():
    """
    deinit multi compilation process
    :return: None
    """
    del_tmp_files(OpCompiler.atc_time_stamp)
    compilers = (OpCompiler.compiler,
                 OpCompiler.autotune_compiler)
    for compiler in compilers:
        if compiler is None:
            continue
        logger.info("destory compiler %s", compiler)
        compiler.destory()
    logger.info("all compiler destoryed")


def get_finished_compilation_task(graph_id):
    """
    return finished compilation task
    :return:
    """
    compiler = OpCompiler.compiler
    if compiler is None:
        return []
    return compiler.get_finished_task(graph_id)


# 'pylint: disable=too-many-arguments
def gen_task_res(ttype, gid, tid, status_code, result, msg, **kwds):
    """
    gen_task_res
    :return: task result
    """
    res = {
        'type': ttype,
        'graph_id': gid,
        'task_id': tid,
        'status_code': status_code,
        'result': result,
        'info_msg': msg
    }

    for key, value in kwds.items():
        res[key] = value

    return res


# pylint: disable=too-many-instance-attributes
class TaskDispatcher:
    """
    Task Dispatcher
    """

    def __init__(self, task_env):
        """
        init
        :param task_env:
        :param worker_list:
        """
        self._task_queue, \
            self.fin_task_queue, \
            self._data_queue, \
            = task_env
        self._data_sync_count = {}
        self._concurrent = 0

    def get_result(self, block=True, queue_idx=-1):
        """
        get result form finished task queue
        :param block:
        :return:
        """
        fin_queue = self.fin_task_queue
        if queue_idx >= 0:
            fin_queue = fin_queue[queue_idx]
        task = fin_queue.get(block)
        self._concurrent -= 1
        return task

    def dispatch(self, task):
        """
        dispatch task to compilation worker
        :param task:
        :return:
        """
        tqueue = self._task_queue
        task.set_data_sync_count(self._data_sync_count)
        tqueue.put(task, True)
        self._concurrent += 1

    def sync_data(self, data_task, pidfrom=0):
        """
        sync data to compilation worker
        :param data_task:
        :return:
        """
        data_task.pidfrom = pidfrom
        for dqueue in self._data_queue:
            dqueue.put(data_task, True)
        count = self._data_sync_count.setdefault(pidfrom, 0)
        self._data_sync_count[pidfrom] = count + 1


# pylint: disable=too-many-instance-attributes
class TaskWorker:
    """
    Task runner
    """

    def __init__(self, task_env):
        """
        init
        :param task_env:
        """
        self._task_queue, \
            self.fin_task_queue, \
            self._data_queue, \
            self.term_event, \
            self._live_checker,\
            self._time_stamp = task_env
        self._block_timeout = 2
        self._data_synced = {}
        self._start = None
        self._end = None
        self._delta = datetime.timedelta()
        self._count = 0

    def do_sync_data(self, block=False, timeout=2):
        """
        load synced data from dispatcher process
        :param block:
        :param timeout:
        :return:
        """
        data_task = self._data_queue.get(block, timeout)
        data_task.run()
        pidfrom = data_task.pidfrom
        count = self._data_synced.setdefault(pidfrom, 0)
        self._data_synced[pidfrom] = count + 1

    def try_sync_data(self):
        """
        try sync data non-blocking
        :return:
        """
        # sync as much as possible
        try:
            while True:
                self.do_sync_data()
        except queue.Empty:
            return

    def mandatory_sync_data(self, need_sync):
        """
        sync exactly count data
        :param count:
        :return:
        """
        # sync exactly 'count' data
        # if there's no enough data, raise exception
        try:
            for pidfrom, count in need_sync.items():
                while count - self._data_synced.get(pidfrom, 0) > 0:
                    self.do_sync_data(True, 60)
        except queue.Empty:
            logger.warning("syncing mandatory data failed. count: %d/%d",
                           count, self._data_synced)

    def loop(self):
        """
        main loop
        :return:
        """
        def _prerun(task):
            """
            set l1 size before run
            """
            cce_policy.set_L1_info("op_L1_space", task.l1size)
            logger.info("set l1 size %s, %s", task.l1size, task)

        def _postrun(task):
            """
            restore l1 size after run
            """
            if task.l1size >= 0:
                cce_policy.set_L1_info("op_L1_space", -1)  # reset l1 space
            logger.info("restore l1 size. %s, %s", task.l1size, task)

        optask_dispatcher = OpCompiler.task_dispatcher
        autotune_dispatcher = OpCompiler.autotune_task_dispatcher
        while not self.term_event.is_set():
            try:
                # check dispatcher process is alive
                if self._live_checker.poll():
                    self._live_checker.recv()

            except EOFError:
                logger.warning("Master process dead. worker process quiting..")
                # Avoid 'Broken PIPE' exception msg of multiprocessing module,
                # we are quiting anyway.
                sys.excepthook = excepthook_silent
                break

            # CAUTION: task dispatcher MUST dispatch data_sync task first
            try:
                task = self._task_queue.get(True, self._block_timeout)
            except queue.Empty:
                task = None

            if task is None:
                self.try_sync_data()
                continue

            if self._start is None:
                self._start = datetime.datetime.now()

            count = task.check_need_sync()
            self.mandatory_sync_data(count)

            _prerun(task)
            if self._time_stamp:
                res = task.run(self._time_stamp)
            else:
                res = task.run()
            _postrun(task)

            self._count += 1
            task_tag = getattr(task, "tag", None)
            if res is None:
                continue

            if task_tag is None:
                self.fin_task_queue.put(res)
            elif task_tag == "autotune":
                logger.info("autotune task %s", task.desc())
                fin_queue = optask_dispatcher.fin_task_queue
                fin_queue.put(res)
            elif task_tag == "autotune_compile_op":
                fin_queue = autotune_dispatcher.fin_task_queue
                fin_queue = fin_queue[task.from_worker]
                fin_queue.put(res)

            self._end = datetime.datetime.now()


class OpTask:
    """
    Base class of various parallel task
    """

    def __init__(self, timeout_ms=2000):
        self._timeout_ms = timeout_ms
        self._data_sync_count = {}
        self.l1size = -1
        self.res = []
        self.pidfrom = 0

    def check_need_sync(self, pidfrom=None):
        """
        check if need to sync data before do this op task
        :return:
        """
        if pidfrom is None:
            return self._data_sync_count

        return self._data_sync_count.get(pidfrom, 0)

    def set_data_sync_count(self, count):
        """
        set the exactly number of data need to sync
        :param count:
        :return:
        """
        self._data_sync_count = count

    def set_l1size(self, l1size):
        """
        set l1 size when compile op
        :param count:
        :return:
        """
        self.l1size = l1size

    def run(self):
        """
        should overide in sub class
        """


class PySysPathTask(OpTask):
    """
    task to add directories to sys.path
    """

    def __init__(self, syspath):
        """
        init
        :param syspath: path needed to add to sys.path
        """
        super().__init__()
        self._syspath = syspath

    def run(self):
        """
        add directory to sys.path
        :return:
        """
        if self._syspath not in sys.path:
            sys.path.append(self._syspath)


class PyImportTask(OpTask):
    """
    task to import py modules
    """

    def __init__(self, module_list):
        """
        init
        :param module_list:
        """
        super().__init__()
        self._module_list = module_list.split(",")

    def run(self):
        """
        do python module import
        :return:
        """
        for mlist in self._module_list:
            if mlist:
                importlib.import_module(mlist)


class ObjSyncTask(OpTask):
    """
    Task to sync module objects from parent to child process.
    """

    def __init__(self, module_name, obj_name, obj_value):
        """
        init
        :param module_name:
        :param obj_name:
        :param obj_value:
        """
        super().__init__()
        self._module_name = module_name
        self._obj_name = obj_name
        self._obj_value = obj_value

    def run(self):
        """
        do the data sync
        :return:
        """
        pymodule = importlib.import_module(self._module_name)
        obj = pickle.loads(zlib.decompress(self._obj_value))
        mysetattr(pymodule, self._obj_name, obj)


class AutotuneTask(OpTask):
    """
    Autotune Task
    """

    def __init__(self, graph_id, task_id, json_str, data_dict, kernel_name):
        """
        init
        :param graph_id:
        :param task_id:
        :param json_str:
        :param kernel_name:
        """
        super().__init__(self)
        self.graph_id = graph_id
        self.task_id = task_id
        self._json_str = json_str
        self._data_dict = data_dict
        self._kernel_name = kernel_name
        self.build_type = 2
        self.tag = "autotune"

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self, time_stamp=None):
        """
        do fusion op compilation
        :return:
        """
        start = datetime.datetime.now()
        try:
            opm = importlib.import_module('auto_tune_main')
            opfunc = mygetattr(opm, 'auto_tune_compile')
            res = opfunc(self._json_str, self._data_dict, time_stamp)
            end = datetime.datetime.now()
            if res:
                infomsg = "auto_tune_compile success. "\
                    "kernel[{}], time:{}/{}"\
                    .format(self._kernel_name, start, end-start)
                return gen_task_res(self.build_type, self.graph_id,
                                    self.task_id, 0, self._kernel_name,
                                    infomsg)
        except Exception as exc:   # 'pylint: disable=broad-except
            if isinstance(getattr(exc, 'args', None), (tuple, list)) and \
               len(exc.args) > 0 and \
               isinstance(exc.args[0], dict) and \
               'errCode' in exc.args[0]:
                except_msg, except_dict_msg = te_log.except_msg()
                errmsg = "autotune op compile fail. kernel_name[{}]"\
                    .format(self._kernel_name)
                logger.info("%s. json:%s\n%s", errmsg,
                            self._json_str, except_msg)
                return gen_task_res(1, self.graph_id,
                                    self.task_id, 1, self._kernel_name,
                                    errmsg,
                                    err_args="json_str:{}".format(
                                        self._json_str),
                                    except_msg=except_msg,
                                    except_tuple_msg=(except_dict_msg,
                                                      self._kernel_name))
            logger.error("auto_tune_compile got unknown error, fallback "
                         "to normal compilation. taskID[%s:%s], "
                         "traceback message: %s",
                         self.graph_id, self.task_id,
                         traceback.format_exc())

        logger.info("autotune compile failed. taskID[%s:%s]",
                    self.graph_id, self.task_id)
        task = FusionOpTask(self.graph_id, self.task_id, self._json_str,
                            self._kernel_name)
        task.set_l1size(self.l1size)
        return task.run()

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "auto_tune_compile",
            "kernel_name": self._kernel_name,
        }
        return json.dumps(op_desc)


class PrebuildTask(OpTask):
    """
    Task to prebuild tbe op
    """

    def __init__(self, graph_id, task_id, op_module,
                 op_type, op_func, *op_args,
                 inputs=None, outputs=None, attrs=None,
                 unknown_shape=False, int64_mode=False):
        """
        init
        :param graph_id:
        :param task_id:
        :param op_module:
        :param op_func:
        :param op_args:
        """
        super().__init__()
        self.graph_id = graph_id
        self.task_id = task_id
        self._op_module = op_module
        self._op_type = op_type
        self._op_func = op_func
        self._op_type = op_type
        self._op_args = op_args
        self._op_inputs = inputs
        self._op_outputs = outputs
        self._op_attrs = attrs
        self.build_type = 0
        self._unknown_shape = unknown_shape
        self._int64_mode = int64_mode

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do prebuild
        :return:
        """
        try:
            start = datetime.datetime.now()
            pattern = fusion_manager.build_single_op(self._op_module, self._op_func,
                                                     self._op_type, "prebuild",
                                                     *self._op_args,
                                                     inputs=self._op_inputs,
                                                     outputs=self._op_outputs,
                                                     attrs=self._op_attrs,
                                                     unknown_shape=self._unknown_shape,
                                                     int64_mode=self._int64_mode)
            end = datetime.datetime.now()
            infomsg = "prebuild success. pattern[{}] module[{}] "\
                "func[{}], time:{}/{}".format(pattern, self._op_module,
                                              self._op_func, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, pattern, infomsg)
        except Exception:       # 'pylint: disable=broad-except
            except_msg, except_dict_msg = te_log.except_msg()
            errmsg = "prebuild failed. module[{}] func[{}]"\
                .format(self._op_module, self._op_func)
            logger.info("%s, args:%s\n%s", errmsg, self._op_args, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, 'None', errmsg,
                                err_args="args:{}, input:{}, outputs:{}, attrs:{}"
                                .format(self._op_args,
                                        self._op_inputs,
                                        self._op_outputs,
                                        self._op_attrs),
                                except_msg=except_msg,
                                except_tuple_msg=(except_dict_msg,
                                                  self._op_func))

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "prebuild",
            "module": self._op_module,
            "args": self._op_args
        }
        return json.dumps(op_desc)


class SingleOpTask(OpTask):
    """
    Task to compile single tbe op
    """
    # pylint: disable=too-many-arguments

    def __init__(self, graph_id, task_id, op_module, op_type,
                 op_func, kernel_name, *op_args,
                 inputs=None, outputs=None, attrs=None,
                 unknown_shape=False, int64_mode=False):
        """
        init
        :param graph_id:
        :param task_id:
        :param op_module:
        :param op_func:
        :param kernel_name:
        :param op_args:
        """
        super().__init__()
        self.graph_id = graph_id
        self.task_id = task_id
        self._op_module = op_module
        self._op_type = op_type
        self._op_func = op_func
        self._op_type = op_type
        self._kernel_name = kernel_name
        self._op_args = op_args
        self._op_inputs = inputs
        self._op_outputs = outputs
        self._op_attrs = attrs
        self.build_type = 1
        self._unknown_shape = unknown_shape
        self._int64mode = int64_mode

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do single op compilation
        :return:
        """
        try:
            start = datetime.datetime.now()
            res = fusion_manager.build_single_op(self._op_module, self._op_func,
                                                 self._op_type, "build",
                                                 *self._op_args,
                                                 inputs=self._op_inputs,
                                                 outputs=self._op_outputs,
                                                 attrs=self._op_attrs,
                                                 unknown_shape=self._unknown_shape,
                                                 int64_mode=self._int64mode)
            end = datetime.datetime.now()
            infomsg = "single op compile success. kernel[{}] "\
                "module[{}] func[{}], time:{}/{}"\
                .format(self._kernel_name, self._op_module,
                        self._op_func, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, self._kernel_name, infomsg, op_res=res)
        except Exception:       # 'pylint: disable=broad-except
            except_msg, except_dict_msg = te_log.except_msg()
            errmsg = "single op compile failed. kernel[{}] "\
                "module[{}] func[{}]"\
                .format(self._kernel_name, self._op_module,
                        self._op_func)
            logger.info("%s, args:%s\n%s", errmsg, self._op_args, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, self._kernel_name, errmsg,
                                err_args="args:{}, input:{}, outputs:{}, attrs:{}"
                                .format(self._op_args,
                                        self._op_inputs,
                                        self._op_outputs,
                                        self._op_attrs),
                                except_msg=except_msg,
                                except_tuple_msg=(except_dict_msg,
                                                  self._op_func))

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "single build",
            "module": self._op_module,
            "kernel_name": self._kernel_name,
            "args": self._op_args
        }
        return json.dumps(op_desc)


class FusionOpTask(OpTask):
    """
    Task to compile fusion op
    """

    def __init__(self, graph_id, task_id, json_str, kernel_name):
        """
        init
        :param graph_id:
        :param task_id:
        :param json_str:
        :param kernel_name:
        """
        super().__init__(self)
        self.graph_id = graph_id
        self.task_id = task_id
        self._json_str = json_str
        self._kernel_name = kernel_name
        self.build_type = 2
        self.tag = None
        self.op_env_cfg = None
        self.from_worker = -1

    def __str__(self):
        """
        string representation
        :return:
        """
        return "taskID[{}.{}]".format(self.graph_id, self.task_id)

    def run(self):
        """
        do fusion op compilation
        :return:
        """
        try:
            start = datetime.datetime.now()
            opm = importlib.import_module("te.platform.fusion_util")
            opfunc = getattr(opm, "fusion_op")
            fusion_manager.op_build_cfg_en()
            op_run_env_func(self.op_env_cfg, "prerun")
            res = opfunc(self._json_str)
            post_res = op_run_env_func(self.op_env_cfg, "postrun")
            end = datetime.datetime.now()
            infomsg = "fusion op compile success. "\
                "kernel[{}], time:{}/{}"\
                .format(self._kernel_name, start, end-start)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                0, self._kernel_name, infomsg, op_res=res,
                                post_res=post_res)
        except Exception:       # 'pylint: disable=broad-except
            except_msg, except_dict_msg = te_log.except_msg()
            errmsg = "fusion op compile fail. kernel_name[{}]"\
                .format(self._kernel_name)
            logger.info("%s. json:%s\n%s", errmsg, self._json_str, except_msg)
            return gen_task_res(self.build_type, self.graph_id, self.task_id,
                                1, self._kernel_name, errmsg,
                                err_args="json_str:{}".format(self._json_str),
                                except_msg=except_msg,
                                except_tuple_msg=(except_dict_msg,
                                                  self._kernel_name))

    def desc(self):
        """
        task description in json format
        """
        op_desc = {
            "type:": "fusion build",
            "kernel_name": self._kernel_name,
        }
        return json.dumps(op_desc)


# 'pylint: disable=too-many-arguments
def dispatch_prebuild_task(graph_id, task_id, l1size,
                           op_module, op_type, op_func,
                           unknown_shape, op_args, int64_mode=False):
    """
    prebuild task
    :param graph_id:
    :param task_id:
    :param op_module:
    :param op_func:
    :param op_args:
    """
    if OpCompiler.compiler is None:
        return
    inputs, outputs, attrs = op_args
    task = PrebuildTask(graph_id, task_id, op_module,
                        op_type, op_func,
                        inputs=inputs, outputs=outputs, attrs=attrs,
                        unknown_shape=unknown_shape,
                        int64_mode=int64_mode)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


# 'pylint: disable=too-many-arguments
def dispatch_single_op_compile_task(graph_id, task_id, l1size, op_module,
                                    op_type, op_func, kernel_name,
                                    unknown_shape, op_args, int64_mode=False):
    """
    single op build task
    :param graph_id:
    :param task_id:
    :param op_module:
    :param op_func:
    :param kernel_name:
    :param op_args:
    """

    if OpCompiler.compiler is None:
        return

    inputs, outputs, attrs = op_args
    task = SingleOpTask(graph_id, task_id, op_module,
                        op_type, op_func, kernel_name,
                        inputs=inputs, outputs=outputs, attrs=attrs,
                        unknown_shape=unknown_shape, int64_mode=int64_mode)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


def dispatch_fusion_op_compile_task(graph_id, task_id, l1size,
                                    json_str, kernel_name):
    """
    fusion op build task
    :param graph_id:
    :param task_id:
    :param json_str:
    :param kernel_name:
    """
    if OpCompiler.compiler is None:
        return
    task = FusionOpTask(graph_id, task_id, json_str, kernel_name)
    task.set_l1size(l1size)
    OpCompiler.compiler.dispatch_task(task)


def dispatch_autotune_task(graph_id, task_id, l1size,
                           json_str, data_dict, kernel_name):
    """
    fusion op build task
    :param graph_id:
    :param task_id:
    :param json_str:
    :param kernel_name:
    """
    if OpCompiler.autotune_compiler is None:
        return
    task = AutotuneTask(graph_id, task_id, json_str, data_dict, kernel_name)
    task.set_l1size(l1size)
    OpCompiler.autotune_compiler.dispatch_task(task)
    update_running_task(task)


def import_py_module(module_list):
    """
    import py module task
    :param module_list:
    """
    if OpCompiler.compiler is None:
        return
    task = PyImportTask(module_list)
    OpCompiler.compiler.sync_data(task)


def sync_py_object(module_name, obj_name, has_value=False, obj_value=None):
    """
    sync python object to worker process
    :param module_name:
    :param obj_name:
    """
    if OpCompiler.master_pid == 0:
        return
    if has_value:
        obj_value = zlib.compress(pickle.dumps(obj_value))
        task = ObjSyncTask(module_name, obj_name, obj_value)
    else:
        opm = importlib.import_module(module_name)
        obj = mygetattr(opm, obj_name)
        obj = zlib.compress(pickle.dumps(obj))
        task = ObjSyncTask(module_name, obj_name, obj)

    if OpCompiler.master_pid == os.getpid():
        OpCompiler.compiler.sync_data(task)
    else:
        # This is autotune sub process
        pid = os.getpid()
        OpCompiler.task_dispatcher.sync_data(task, pidfrom=pid)


def sync_py_object_serial(module_name, obj_name, obj_value):
    """
    sync python object
    :param module_name:
    :param obj_name:
    :param obj_value:
    """
    pymodule = importlib.import_module(module_name)
    mysetattr(pymodule, obj_name, obj_value)


def sync_syspath(syspath):
    """
    sync syspath to worker process
    :param syspath: the path needed to add to sys.path of worker process
    """
    if OpCompiler.compiler is None:
        return
    task = PySysPathTask(syspath)
    OpCompiler.compiler.sync_data(task)


def op_run_env_func(op_env_cfg, func_name):
    """
    run op environment setting function
    """
    if op_env_cfg is None:
        return
    prerun = op_env_cfg.get(func_name)
    if prerun is None:
        return
    try:
        args = []
        kwargs = {}
        if len(prerun) > 1:
            args = prerun[1]
        if len(prerun) > 2:
            kwargs = prerun[2]

        return prerun[0](*args, **kwargs)
    except Exception:   # 'pylint: disable=broad-except
        except_msg, _ = te_log.except_msg()
        logger.info("%s", except_msg)
    return


def compile_op(json_str, op_env_cfg=None):
    """
    compile op parallelly
    """
    op_desc = json.loads(json_str)
    kernel_name = op_desc["fusion_op_name"]
    gid = os.getpid()
    tid = Counter.next()

    if OpCompiler.master_pid == 0:
        # parallel compiler not active
        fusion_manager.op_build_cfg_en()
        try:
            op_run_env_func(op_env_cfg, "prerun")
            fusion_util.fusion_op(json_str)
            post_res = op_run_env_func(op_env_cfg, "postrun")
            res = gen_task_res(2, gid, tid, 0, kernel_name,
                               "syncbuild succ", post_res=post_res)
        except Exception:   # 'pylint: disable=broad-except
            except_msg, _ = te_log.except_msg()
            errmsg = "compile op fail. kernel[{}]".format(kernel_name)
            logger.info("%s. json:%s\n%s", errmsg, json_str, except_msg)
            res = gen_task_res(2, gid, tid, 1, kernel_name,
                               "syncbuild faild")
        return DeferredOpRes(gid, tid, res)

    task = FusionOpTask(gid, tid, json_str, op_desc['fusion_op_name'])
    task.op_env_cfg = op_env_cfg

    if OpCompiler.master_pid == gid:
        # call from parent process
        OpCompiler.compiler.dispatch_task(task)
        return DeferredOpRes(gid, tid)

    # call from autotune sub process
    task.tag = "autotune_compile_op"
    task.from_worker = OpCompiler.autotune_worker_idx
    logger.info("autotune_compile_op from %s, pid %s",
                task.from_worker, gid)
    OpCompiler.task_dispatcher.dispatch(task)
    return DeferredOpRes(gid, tid, tag=task.tag, from_worker=task.from_worker)


def compile_op_sync(json_str):
    """
    compile op synchronously
    """
    defer = compile_op(json_str)
    while True:
        time.sleep(0.01)
        res = defer.get()
        if res is not None:
            return


def update_running_task(task):
    """
    update runing task status
    """
    if OpCompiler.compiler is None:
        return
    OpCompiler.compiler.update_running_task(task)
