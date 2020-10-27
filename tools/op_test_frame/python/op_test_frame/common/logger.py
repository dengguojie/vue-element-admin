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

import inspect
from datetime import datetime

DEBUG = "DEBUG"
INFO = "INFO"
WARN = "WARN"
ERROR = "ERROR"


def log(level, file, line, msg):
    def _get_time_str():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    print("[%s] %s [File \"%s\", line %d] %s" % (level, _get_time_str(), file, line, msg))


def log_warn(msg):
    caller = inspect.stack()[1]
    log(WARN, caller.filename, caller.lineno, msg)


def log_debug(msg):
    caller = inspect.stack()[1]
    log(DEBUG, caller.filename, caller.lineno, msg)


def log_info(msg):
    caller = inspect.stack()[1]
    log(INFO, caller.filename, caller.lineno, msg)


def log_err(msg):
    caller = inspect.stack()[1]
    log(ERROR, caller.filename, caller.lineno, msg)
