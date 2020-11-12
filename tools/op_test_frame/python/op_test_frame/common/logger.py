# Copyright 2020 Huawei Technologies Co., Ltd
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
logger
"""
import inspect
from datetime import datetime

DEBUG = "DEBUG"
INFO = "INFO"
WARN = "WARN"
ERROR = "ERROR"

LOG_LEVEL = "INFO"


def set_logger_level(level):
    """
    set logger level
    :param level: level
    :return: None
    """
    global LOG_LEVEL  # pylint: disable=global-statement
    LOG_LEVEL = level


def log(level, file, line, msg):
    """
    print log
    :param level: level
    :param file: file
    :param line: line
    :param msg: msg
    :return: None
    """

    def _get_time_str():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    print("[%s] %s [File \"%s\", line %d] %s" % (level, _get_time_str(), file, line, msg))


def log_warn(msg):
    """
    log warn
    :param msg: log msg
    :return: None
    """
    caller = inspect.stack()[1]
    log(WARN, caller.filename, caller.lineno, msg)


def log_debug(msg):
    """
    log debug
    :param msg: log msg
    :return: None
    """
    if LOG_LEVEL not in ("DEBUG",):
        return
    caller = inspect.stack()[1]
    log(DEBUG, caller.filename, caller.lineno, msg)


def log_info(msg):
    """
    log info
    :param msg: log msg
    :return: None
    """
    if LOG_LEVEL not in ("DEBUG", "INFO"):
        return
    caller = inspect.stack()[1]
    log(INFO, caller.filename, caller.lineno, msg)


def log_err(msg):
    """
    log err
    :param msg: log msg
    :return: None
    """
    caller = inspect.stack()[1]
    log(ERROR, caller.filename, caller.lineno, msg)
