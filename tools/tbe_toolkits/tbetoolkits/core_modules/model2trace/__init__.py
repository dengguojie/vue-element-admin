#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Parse pem model dump to chrome trace
"""

# Standard Packages
import time
from pathlib import Path

# Third-party Packages
from .Ascend910_CA import Ascend910_CA_ModelParser
from .Ascend910_PEM import Ascend910_PEM_ModelParser
from .Hi3796CV300CS_CA import Hi3796CV300CS_CA_ModelParser
from .common import BaseModelParser
from .classes import ChromeTraceJson
from ...utilities import MODE


registry = {
    ("Ascend910A", MODE.ASCEND_PEMMODEL): Ascend910_PEM_ModelParser,
    ("Ascend910A", MODE.ASCEND_CAMODEL): Ascend910_CA_ModelParser,
    ("Hi3796CV300CS", MODE.ASCEND_CAMODEL): Hi3796CV300CS_CA_ModelParser,
    ("Ascend710", MODE.ASCEND_CAMODEL): Hi3796CV300CS_CA_ModelParser
}


def get_registered_module(platform: str, model_type: MODE):
    pair = (platform, model_type)
    if pair in registry:
        return registry[pair]
    else:
        raise NotImplementedError(f"{pair} parsing module is not implemented yet, "
                                  f"currently supported pairs are {registry.keys()}")


def parse_dumps_in_folder(folder_path: str = ".", core_index: int = 0,
                          platform: str = "Ascend910A",
                          model_type: MODE = MODE.ASCEND_PEMMODEL) -> ChromeTraceJson:
    parse_folder: Path = Path(folder_path)
    if not parse_folder.exists():
        raise RuntimeError(f"Could not parse model dumps in folder {folder_path}, path does not exist!")
    container = ChromeTraceJson()
    parser = get_registered_module(platform, model_type)(container)
    # ICache Parse
    before_icache = time.time()
    icache_logs = parser.read_dumps(parse_folder, core_index=core_index)
    if icache_logs:
        parser.initialize_metadata()
    parser.handle_icache_log(icache_logs)
    print("ICache parsing costs", time.time() - before_icache)
    # Instr Parse
    before_instr = time.time()
    raw_instr_logs = parser.read_dumps(parse_folder, dump_type="instr", core_index=core_index)
    raw_instr_popped_logs = parser.read_dumps(parse_folder, dump_type="instr_popped", core_index=core_index)
    parser.handle_instr_log(raw_instr_logs, raw_instr_popped_logs)
    print("Instruction parsing costs", time.time() - before_instr)
    return container
