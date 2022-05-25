import importlib
import logging
import os
import sys
from functools import wraps
from typing import List, Sequence


class Plugin:
    """Single plugin"""

    def __init__(self, name) -> None:
        self._dict = {}
        self._name = name

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("op [%s] is not registered for %s functions", key, self._name)
            raise e

    def __setitem__(self, key, value):
        if key in self._dict:
            logging.warning("%s function for op [%s] has already been registered!", self._name, key)
        self._dict[key] = value

    def register(self, op_names: Sequence[str]):
        """Register plugin function
           use it like: @Plugins.golden.register(["op"])
        """
        if not isinstance(op_names, (list, tuple)):
            raise TypeError("Register function for %s functions must receive a list or tuple, "
                            "rather than %s" % (self._name, str(op_names)))

        def __inner_registry(func):
            for op in op_names:
                self._dict[op] = func
            return func

        return __inner_registry


class Plugins:
    """All plugins"""

    golden = Plugin("golden")
    input = Plugin("input")

    def __init__(self) -> None:
        raise RuntimeError("Class [Plugins] can not be used as an instance.")

    @classmethod
    def get_all(cls):
        all = {}
        for k, v in cls.__dict__.items():
            if isinstance(v, Plugin):
                all[k] = v
        return all


def import_all_plugins(files: List[str]) -> None:
    """Import all modules dynamically"""
    for f in files:
        if not f.endswith(".py"):
            continue
        plugin_realpath = os.path.realpath(f)
        plugin_dir, plugin_file = os.path.split(plugin_realpath)
        plugin_name = os.path.splitext(plugin_file)[0]
        if plugin_dir not in sys.path:
            sys.path.append(plugin_dir)
        try:
            importlib.import_module(plugin_name)
            logging.debug("plugin [%s] loaded.", plugin_name)
        except ImportError as e:
            logging.error("plugin [%s] load failed.", plugin_realpath)
            raise e