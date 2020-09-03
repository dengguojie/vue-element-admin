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

__version__ = "0.1"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


with open("MANIFEST.in", "w") as fo:
    fo.write("include libs/libmodel_run_tool.so\n")

setup_kwargs = {
    "include_package_data": True
}

def read_txt(file_name):
    return open(file_name).read()

setup(
    name="op_test_frame",
    version=__version__,
    scripts=['op_test_frame/scripts/op_ut_run', 'op_test_frame/scripts/op_ut_helper'],
    zip_safe=False,
    packages=find_packages(),
    install_requires=[],
    license=read_txt("LICENSE"),
    **{
        "include_package_data": True,
        "data_files": [('lib', ['op_test_frame/libs/libmodel_run_tool.so']),
		       ('output_dir', ['op_test_frame/st/template/acl_op_src/CMakeLists.txt']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/inc/common.h']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/inc/op_execute.h']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/inc/op_runner.h']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/inc/op_test_desc.h']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/inc/op_test.h']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/run/out/test_data/config/acl.json']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/run/out/test_data/config/acl_op.json']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/CMakeLists.txt']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/common.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/main.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/op_execute.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/op_runner.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/op_test.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/op_test_desc.cpp']),
                       ('output_dir', ['op_test_frame/st/template/acl_op_src/src/testcase.cpp'])]
    }

)
