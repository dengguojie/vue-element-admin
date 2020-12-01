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
file util module
"""
import os


def _mkdir_without_file_exist_err(dir_path, mode):
    try:
        os.mkdir(dir_path, mode)
    except FileExistsError as err:
        pass
    except:
        raise


def makedirs(path, mode):
    """
    like sheel makedir
    :param path: dirs path
    :param mode: dir mode
    :return: None
    """

    def _rec_makedir(dir_path):
        parent_dir = os.path.dirname(dir_path)
        if parent_dir == dir_path:
            # root dir, not need make
            return
        if not os.path.exists(parent_dir):
            _rec_makedir(parent_dir)
            _mkdir_without_file_exist_err(dir_path, mode)
        else:
            _mkdir_without_file_exist_err(dir_path, mode)

    path = os.path.realpath(path)
    _rec_makedir(path)
