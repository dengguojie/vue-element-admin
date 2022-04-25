#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Fetch models
"""
# Standard Packages
import os
import pathlib
import logging
import platform
import shutil

# Third-party Packages
import tbetoolkits
from tbetoolkits.utilities import MODE
from tbetoolkits.utilities import download_file


def get_models(_platform: str, mode: MODE, unzip_config: bool = False):
    """
    Get model packages
    :param _platform:
    :param mode:
    :param unzip_config:
    :return:
    """
    raise NotImplementedError()