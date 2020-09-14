#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

TBE auto_tiling_log
"""
import os

class AutoTilingLog():
    """
    log for tiling
    """
    def __init__(self):
        self.use_slog = False
        try:
            from te.utils.AscendLog import LOGGER
            self.slog = LOGGER
            self.use_slog = True
        except ImportError:
            import logging
            self.log = logging.getLogger('auto_tiling')
            self.log.setLevel((int(os.getenv("GLOBAL_LOG_LEVEL", "3")) + 1) * 10)

    def __new__(cls, *args, **kwargs):
        if not hasattr(AutoTilingLog, "_instance"):
            AutoTilingLog._instance = object.__new__(cls)
        return AutoTilingLog._instance

    def info(self, log_msg):
        """
        info level
        """
        if self.use_slog:
            self.slog.info(self.slog.module.tbe, log_msg)
        else:
            self.log.info(log_msg)

    def debug(self, log_msg):
        """
        debug level
        """
        if self.use_slog:
            self.slog.debug(self.slog.module.tbe, log_msg)
        else:
            self.log.debug(log_msg)

    def warn(self, log_msg):
        """
        warn level
        """
        if self.use_slog:
            self.slog.warn(self.slog.module.tbe, log_msg)
        else:
            self.log.warn(log_msg)

    def error(self, log_msg):
        """
        error level
        """
        if self.use_slog:
            self.slog.error(self.slog.module.tbe, log_msg)
        else:
            self.log.error(log_msg)

AUTOTILINGLOG = AutoTilingLog()
