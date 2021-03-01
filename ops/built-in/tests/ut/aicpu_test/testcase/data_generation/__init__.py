"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from os.path import dirname, basename, isfile
import glob
all_files = glob.glob(dirname(__file__) + "/*.py")
all_modules = [basename(f)[:-3] for f in all_files if isfile(f) and not f.endswith('__init__.py')]
modules = [module for module in all_modules if module.endswith('_gen_data')]
__all__ = modules
