#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Precious Synchronization Related Utilities
"""
# Standard Packages
import time
import contextlib


@contextlib.contextmanager
def SelectiveNonBlockingLock(locks: tuple, device_id_pointer: list):
    """
    This Lock select an acquirable lock from tuple of locks
    :param locks:
    :param device_id_pointer:
    :return:
    """
    if len(locks) == 0 or all(x is None for x in locks):
        raise RuntimeError("No available locks, please check your lock config")
    my_lock = None
    while my_lock is None:
        for idx, lock in enumerate(locks):
            if lock is not None and lock.acquire(False):
                my_lock = lock
                device_id_pointer[0] = idx
                break
        time.sleep(0.1)
    try:
        yield
    finally:
        # noinspection PyBroadException
        try:
            my_lock.release()
        except:
            pass
