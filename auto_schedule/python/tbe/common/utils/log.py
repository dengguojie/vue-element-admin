#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
tbe log
"""
import inspect
import logging
import os
import sys

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_DEFAULT_LEVEL = logging.INFO

IS_USE_SLOG = True
try:
    from tbe.common.utils.AscendLog import AscendLog
    S_LOGGER = AscendLog()
except BaseException:
    IS_USE_SLOG = False

    GLOBAL_LOG_ID = "TBE_AUTOSCHEDULE"
    LOGGER = logging.getLogger(GLOBAL_LOG_ID)
    LOGGER.propagate = 0
    LOGGER.setLevel(LOG_DEFAULT_LEVEL)

    STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
    STREAM_HANDLER.setLevel(LOG_DEFAULT_LEVEL)
    LOG_FORMAT = "[%(levelname)s][%(asctime)s]%(message)s"
    STREAM_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))
    LOGGER.addHandler(STREAM_HANDLER)


def info(log_msg, *log_paras):
    """
    info log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras

    if IS_USE_SLOG:
        S_LOGGER.info(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.info(log_all_msg)


def debug(log_msg, *log_paras):
    """
    debug log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras

    if IS_USE_SLOG:
        S_LOGGER.debug(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.debug(log_all_msg)


def warn(log_msg, *log_paras):
    """
    warning log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras

    if IS_USE_SLOG:
        S_LOGGER.warn(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.warning(log_all_msg)


def error(log_msg, *log_paras):
    """
    error log
    :param log_msg:
    :param log_paras:
    """
    line_no = inspect.currentframe().f_back.f_lineno
    funcname = inspect.currentframe().f_back.f_code.co_name
    co_filename = inspect.currentframe().f_back.f_code.co_filename
    filename = os.path.relpath(co_filename, FILE_PATH)
    log_str = '[%s:%d][%s] ' % (filename, line_no, funcname)
    log_all_msg = log_str + log_msg % log_paras

    if IS_USE_SLOG:
        S_LOGGER.error(S_LOGGER.module.tbe, log_all_msg)
    else:
        LOGGER.error(log_all_msg)
