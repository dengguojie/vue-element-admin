#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Main Entrance
"""
# Standard Packages
import os
import sys
import logging

# Third-Party Packages
import tbetoolkits
from tbetoolkits.plugins import import_all_plugins, Plugins


def _load_plugins(csv_files) -> None:
    if not csv_files:
        return
    if isinstance(csv_files, str):
        csv_files = csv_files.split(',')
    py_files = set()
    for cf in csv_files:
        cf_name = os.path.splitext(cf)[0]
        py_file = ''.join([cf_name, "_csv", ".py"])
        if os.path.exists(py_file):
            py_files.add(py_file)
    import_all_plugins(py_files)


def _initialize_switches(args) -> tbetoolkits.utilities.SWITCHES:
    switches = tbetoolkits.utilities.SWITCHES()
    switches.logging_to_file = True
    # Override soc spec such as CORE_NUM
    switches.soc_spec_override = {}
    # Realtime compilation switches, change to False if you want to use manually modified .cce or .o file
    switches.dyn_switches.realtime = True
    switches.stc_switches.realtime = True
    switches.cst_switches.realtime = True
    switches.bin_switches.realtime = True
    # RTS Online profiling switches, change to False if you want to have PMU data on Ascend310P/Ascend615/Ascend610
    switches.dyn_switches.rts_prof = True
    switches.stc_switches.rts_prof = True
    switches.cst_switches.rts_prof = True
    switches.bin_switches.rts_prof = True
    # Dump switch, you can select from NO, INPUT, OUTPUT, GOLDEN, INOUT, INGOLD, OUTGOLD and FULL
    switches.dump_mode = tbetoolkits.utilities.DUMP_LEVEL.NO
    tbetoolkits.utilities.parse_params(switches, args)
    # Load plugins
    _load_plugins(args[0])
    switches.plugins = Plugins.get_all()
    return switches


class Entrance:
    """
    Class wrapper for entrance function
    """

    @staticmethod
    def main():
        """
        Standard main entrance function
        """
        print("Loading tensorflow...")
        # noinspection PyBroadException
        try:
            __import__("tensorflow")
        except Exception:
            print("Tensorflow load failed")
        # Parse Input Parameters
        switches = _initialize_switches(sys.argv[1:])
        tbetoolkits.utilities.set_global_storage(switches)
        logging_to_file = tbetoolkits.utilities.get_global_storage().logging_to_file
        tbetoolkits.core_modules.tbe_logging.default_logging_config(file_handler=logging_to_file)
        # GPU mode branch
        if tbetoolkits.utilities.get_global_storage().mode.is_gpu():
            ins = tbetoolkits.core_modules.gpu.GPUProfilingInstance()
            ins.profile()
        else:
            ins = tbetoolkits.core_modules.profiling.ProfilingInstance()
            ins.profile()


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        Entrance.main()
    except:
        logging.exception("TBEToolkits Main Sequence Failed:")
        for proc in tbetoolkits.core_modules.tbe_multiprocessing.SimpleCommandProcess.all_processes:
            proc.close()
