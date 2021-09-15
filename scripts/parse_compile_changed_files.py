#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import sys

CPU = "CPU"
PASS = "PASS"
TILING = "TILING"
PROTO = "PROTO"
TF_PLUGIN = "TF_PLUGIN"
ONNX_PLUGIN = "ONNX_PLUGIN"
CAFFE_PLUGIN = "CAFFE_PLUGIN"
OTHER_FILE="OTHER_FILE"
SCHEDULE="SCHEDULE"
TBE="TBE"
FUSION_RULES="FUSION_RULES"
TOOLS="TOOLS"
class FileChangeInfo:
    def __init__(self, proto_changed_files=[], tiling_changed_files=[], pass_changed_files=[], aicpu_changed_files=[],
                 plugin_changed_files=[], onnx_plugin_changed_files=[], caffe_plugin_changed_files=[],
                 other_changed_files=[], schedule_changed_file=[], tbe_changed_file=[],
                 fusion_rules_change_file=[], tools_change_file=[]):
        self.proto_changed_files = proto_changed_files
        self.tiling_changed_files = tiling_changed_files
        self.pass_changed_files = pass_changed_files
        self.aicpu_changed_files = aicpu_changed_files
        self.plugin_changed_files = plugin_changed_files
        self.onnx_plugin_changed_files = onnx_plugin_changed_files
        self.caffe_plugin_changed_files = caffe_plugin_changed_files
        self.other_changed_files = other_changed_files
        self.schedule_changed_file = schedule_changed_file
        self.tbe_changed_file = tbe_changed_file
        self.fusion_rules_change_file = fusion_rules_change_file
        self.tools_change_file = tools_change_file

    def print_change_info(self):
        print("=========================================================================\n")
        print("changed file info\n")
        print("-------------------------------------------------------------------------\n")
        print("infershape proto changed files: \n%s" % "\n".join(self.proto_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("tiling changed files: \n%s" % "\n".join(self.tiling_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("fusion changed files: \n%s" % "\n".join(self.pass_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("aicpu changed files: \n%s" % "\n".join(self.aicpu_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("plugin changed files: \n%s" % "\n".join(self.plugin_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("onnx plugin changed files: \n%s" % "\n".join(self.onnx_plugin_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("caffe plugin changed files: \n%s" % "\n".join(self.caffe_plugin_changed_files))
        print("-------------------------------------------------------------------------\n")
        print("other changed files: \n%s" % "\n".join(self.other_changed_files))
        print("=========================================================================\n")
        print("schedule changed files: \n%s" % "\n".join(self.schedule_changed_file))
        print("=========================================================================\n")
        print("tbe changed files: \n%s" % "\n".join(self.tbe_changed_file))
        print("=========================================================================\n")
        print("fusion_rules changed files: \n%s" % "\n".join(self.fusion_rules_change_file))
        print("=========================================================================\n")
        print("tools changed files: \n%s" % "\n".join(self.tools_change_file))
        print("=========================================================================\n")


def get_file_change_info_from_ci(changed_file_info_from_ci):
    """
      get file change info from ci, ci will write `git diff > /or_filelist.txt`
      :param changed_file_info_from_ci: git diff result file from ci
      :return: None or FileChangeInf
      """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        print("[ERROR] change file is not exist, can not get file change info in this pull request.")
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        proto_changed_files = []
        tiling_changed_files = []
        pass_changed_files = []
        aicpu_changed_files = []
        plugin_changed_files = []
        onnx_plugin_changed_files = []
        caffe_plugin_changed_files = []
        other_changed_files = []
        schedule_changed_file = []
        tbe_changed_file = []
        fusion_rules_change_file = []
        tools_change_file = []

        base_path = os.path.join("ops", "built-in")
        not_compile_file = ["OWNERS", "NOTICE", "LICENSE", "README.md", "classify_rule.yaml" ]

        for line in lines:
            line = line.strip()
            if line.startswith("auto_schedule"):
                schedule_changed_file.append(line)  
                continue
            if line.startswith(os.path.join(base_path, "tbe")):
                tbe_changed_file.append(line)
                continue
            if line.startswith(os.path.join(base_path, "fusion_rules")):
                fusion_rules_change_file.append(line)
                continue
            if line.startswith("tools"):
                tools_change_file.append(line)
                continue
            if line.startswith("scripts"):
                continue
            if line.endswith(".py"):
                continue
            if line.startswith(os.path.join(base_path, "tests")):
                continue
            if line in not_compile_file:
                continue

            if line.startswith(os.path.join(base_path, "aicpu")):
                aicpu_changed_files.append(line)
            elif line.startswith(os.path.join(base_path, "fusion_pass")):
                pass_changed_files.append(line)
            elif line.startswith(os.path.join(base_path, "op_proto")):
                proto_changed_files.append(line)
            elif line.startswith(os.path.join(base_path, "op_tiling")):
                tiling_changed_files.append(lines)
            elif line.startswith(os.path.join(base_path, "framework", "tf_plugin")) :
                plugin_changed_files.append(line)
            elif line.startswith(os.path.join(base_path, "framework", "onnx_plugin")):
                onnx_plugin_changed_files.append(line)
            elif line.startswith(os.path.join(base_path, "framework", "caffe_plugin")):
                caffe_plugin_changed_files.append(line)
            else:
                other_changed_files.append(line)
    return FileChangeInfo(proto_changed_files=proto_changed_files, tiling_changed_files=tiling_changed_files,
                          pass_changed_files=pass_changed_files, aicpu_changed_files=aicpu_changed_files,
                          plugin_changed_files=plugin_changed_files,
                          onnx_plugin_changed_files=onnx_plugin_changed_files, caffe_plugin_changed_files=caffe_plugin_changed_files,
                          other_changed_files=other_changed_files, schedule_changed_file=schedule_changed_file,
                          tbe_changed_file=tbe_changed_file,
                          fusion_rules_change_file=fusion_rules_change_file, tools_change_file=tools_change_file)


def get_change_relate_dir_list(changed_file_info_from_ci):
    file_change_info = get_file_change_info_from_ci(
        changed_file_info_from_ci)
    if not file_change_info:
        print("[INFO] not found file change info, run all c++.")
        return None
    # file_change_info.print_change_info()

    def _get_relate_list_by_file_change():

        relate = set()
        other_file = set()
        if len(file_change_info.aicpu_changed_files) > 0:
            relate.add(CPU)
        if len(file_change_info.pass_changed_files) > 0:
            relate.add(PASS)
        if len(file_change_info.tiling_changed_files) > 0:
            relate.add(TILING)
        if len(file_change_info.proto_changed_files) > 0:
            relate.add(PROTO)
        if len(file_change_info.plugin_changed_files) > 0:
            relate.add(TF_PLUGIN)
        if len(file_change_info.onnx_plugin_changed_files) > 0:
            relate.add(ONNX_PLUGIN)
        if len(file_change_info.caffe_plugin_changed_files) > 0:
            relate.add(CAFFE_PLUGIN)
        
        if len(file_change_info.other_changed_files) > 0:
            other_file.add(OTHER_FILE)
        if len(file_change_info.schedule_changed_file) > 0:
            relate.add(SCHEDULE)
        if len(file_change_info.tbe_changed_file) > 0:
            relate.add(TBE)
        if len(file_change_info.fusion_rules_change_file) > 0:
            relate.add(FUSION_RULES)
        if len(file_change_info.tools_change_file) > 0:
            relate.add(TOOLS)
        return relate,other_file

    try:
        relates,other_file = _get_relate_list_by_file_change()
    except BaseException as e:
        print(e.args)
        return None
    return str(relates),str(other_file)


if __name__ == '__main__':
  print(get_change_relate_dir_list(sys.argv[1]))
