# Copyright 2020 Huawei Technologies Co., Ltd
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
Function:
This file mainly set up for op_gen whl package.
"""

import os
from setuptools import setup
os.environ['SOURCE_DATE_EPOCH'] = str(int(os.path.getctime(os.path.realpath(__file__))))

__version__ = "0.1"


with open("MANIFEST.in", "w") as fo:
    fo.write("recursive-include ../op_gen/template *\n"
             "recursive-include ../op_gen/json_template *\n"
             "recursive-include ../op_gen/config *\n")

setup_kwargs = {
    "include_package_data": True
}


def read_txt(file_name):
    """
    read from file
    """
    return open(file_name).read()


setup(
    name="op_gen",
    version=__version__,
    scripts=['./msopgen'],
    zip_safe=False,
    package_dir={'': '../'},
    packages=['op_gen', 'op_gen/interface'],
    install_requires=[],
    license=read_txt("LICENSE"),
    include_package_data=True
)
