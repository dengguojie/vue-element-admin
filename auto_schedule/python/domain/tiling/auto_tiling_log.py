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
TBE auto_tiling_log
"""
import os
LOG_LEVEL = 10

class AutoTilingLog():
    """
    log for tiling
    """

    def __init__(self):
        """
        init the specific object
        """
        self.use_slog = False
        try:
            from te.utils.AscendLog import LOGGER
            self.slog = LOGGER
            self.use_slog = True
        except ImportError:
            # for ut/st, AscendLog is different path in source and package
            import logging
            self.log = logging.getLogger('auto_tiling')
            self.log.setLevel((int(os.getenv("GLOBAL_LOG_LEVEL", "3")) + 1) * LOG_LEVEL)

    def __new__(cls, *args, **kwargs):
        """
        create new object
        """
        if not hasattr(AutoTilingLog, "_instance"):
            AutoTilingLog._instance = object.__new__(cls)
        return AutoTilingLog._instance

    def info(self, log_msg):
        """
        algorithm: info
        info level log

        parameters
        ----------
        log_msg:str
          info message

        Returns
        ---------
        None
        """
        if self.use_slog:
            self.slog.info(self.slog.module.tbe, log_msg)
        else:
            self.log.info(log_msg)

    def debug(self, log_msg):
        """
        algorithm: debug
        debug level log

        parameters
        ----------
        log_msg:str
          debug message

        Returns
        ---------
        None
        """
        if self.use_slog:
            self.slog.debug(self.slog.module.tbe, log_msg)
        else:
            self.log.debug(log_msg)

    def warn(self, log_msg):
        """
        algorithm: warn
        warn level log

        parameters
        ----------
        log_msg:str
          warn message

        Returns
        ---------
        None
        """
        if self.use_slog:
            self.slog.warn(self.slog.module.tbe, log_msg)
        else:
            self.log.warn(log_msg)

    def error(self, log_msg):
        """
        algorithm: error
        error level log

        parameters
        ----------
        log_msg:str
          error message

        Returns
        ---------
        None
        """
        if self.use_slog:
            self.slog.error(self.slog.module.tbe, log_msg)
        else:
            self.log.error(log_msg)


AUTOTILINGLOG = AutoTilingLog()
