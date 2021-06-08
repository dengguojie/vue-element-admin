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

from setuptools import setup
from setuptools import find_packages
from setuptools.dist import Distribution
import os
os.environ['SOURCE_DATE_EPOCH'] = str(int(os.path.getctime(os.path.realpath(__file__))))

__version__ = "0.1"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


with open("MANIFEST.in", "w") as fo:
    fo.write("recursive-include op_test_frame/st/template *\n"
             "include op_test_frame/st/interface/framework/framework.json\n"
             "include op_test_frame/st/config/white_list_config.json\n")

setup_kwargs = {
    "include_package_data": True
}


def read_txt(file_name):
    return open(file_name).read()

setup(
    name="op_test_frame",
    version=__version__,
    scripts=['op_test_frame/scripts/op_ut_run',
             'op_test_frame/scripts/op_ut_helper',
             'op_test_frame/scripts/msopst',
             'op_test_frame/scripts/msopst.ini'],
    zip_safe=False,
    packages=find_packages(),
    install_requires=[],
    distclass=BinaryDistribution,
    license=read_txt("LICENSE"),
    ** {
        "include_package_data": True
    }
)
