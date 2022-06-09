#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Knowledge Base Sequence for Universal testcases
"""
# Standard Packages
import time
# Third-party Packages
from ..operator.op_interface import OperatorInterface
from ..tbe_multiprocessing.pool import get_process_context


def knowledge_base_sequence():
    interface = OperatorInterface()
    with interface.knowledge_base():
        while get_process_context().get_data("switch"):
            time.sleep(1)