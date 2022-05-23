#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Knowledge Base Sequence for Universal testcases
"""
# Standard Packages
import time
# Third-party Packages
from ..operator.op_interface import OperatorInterface


def knowledge_base_sequence():
    interface = OperatorInterface()
    with interface.knowledge_base():
        while True:
            time.sleep(1)