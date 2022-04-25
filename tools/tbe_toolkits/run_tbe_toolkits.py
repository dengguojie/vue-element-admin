#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Main Entrance
"""
# Standard Packages
import sys
import logging

# Third-Party Packages
import tbetoolkits


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
    # RTS Online profiling switches, change to False if you want to have PMU data on Ascend710/615/610
    switches.dyn_switches.rts_prof = True
    switches.stc_switches.rts_prof = True
    switches.cst_switches.rts_prof = True
    switches.bin_switches.rts_prof = True
    # Dump switch, you can select from NO, INPUT, OUTPUT, GOLDEN, INOUT, INGOLD, OUTGOLD and FULL
    switches.dump_mode = tbetoolkits.utilities.DUMP_LEVEL.NO
    tbetoolkits.utilities.parse_params(switches, args)
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
