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

set(SAMPLES_INSTALL_PATH "${INSTALL_PATH}/atc/sample/op")

# install all to atc/sample/op/all
set(_all_path "${CMAKE_CURRENT_SOURCE_DIR}/all")
set(_all_dest "${SAMPLES_INSTALL_PATH}/all")
add_custom_target(copy_all_op_samples ALL)
cann_install(
  TARGET      copy_all_op_samples
  FILES       ${_all_path}/build.sh
              ${_all_path}/CMakeLists.txt
              ${_all_path}/custom.proto
              ${_all_path}/.project
              ${_all_path}/README.md
  DIRECTORIES ${_all_path}/cmake
              ${_all_path}/cpukernel
              ${_all_path}/framework
              ${_all_path}/op_proto
              ${_all_path}/scripts
              ${_all_path}/tools
  DESTINATION ${_all_dest}
)
cann_install(
  TARGET      copy_all_op_samples
  FILES       ${_all_path}/tbe/CMakeLists.txt
  DIRECTORIES ${_all_path}/tbe/impl
              ${_all_path}/tbe/op_info_cfg
              ${_all_path}/tbe/testcases/tf_test
  DESTINATION ${_all_dest}/tbe
)


# install tensorflow to atc/sample/op/tensorflow
set(_tf_path "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow")
set(_tf_dest "${SAMPLES_INSTALL_PATH}/tensorflow")
add_custom_target(copy_tf_op_samples ALL)
cann_install(
  TARGET      copy_tf_op_samples
  FILES       ${_tf_path}/CMakeLists.txt
              ${_tf_path}/.project
              ${_tf_path}/README.md
  DIRECTORIES ${_tf_path}/cmake
              ${_tf_path}/framework
              ${_tf_path}/op_proto
              ${_tf_path}/scripts
              ${_tf_path}/tools
  DESTINATION ${_tf_dest}
)
cann_install(
  TARGET      copy_tf_op_samples
  FILES       ${_tf_path}/tbe/CMakeLists.txt
  DIRECTORIES ${_tf_path}/tbe/impl
              ${_tf_path}/tbe/op_info_cfg
              ${_tf_path}/tbe/testcases/tf_test
  DESTINATION ${_tf_dest}/tbe
)


# install caffe to atc/sample/op/caffe
set(_caffe_path "${CMAKE_CURRENT_SOURCE_DIR}/caffe")
set(_caffe_dest "${SAMPLES_INSTALL_PATH}/caffe")
add_custom_target(copy_caffe_op_samples ALL)
cann_install(
  TARGET      copy_caffe_op_samples
  FILES       ${_caffe_path}/build.sh
              ${_caffe_path}/CMakeLists.txt
              ${_caffe_path}/custom.proto
              ${_caffe_path}/.project
              ${_caffe_path}/README.md
  DIRECTORIES ${_caffe_path}/cmake
              ${_caffe_path}/framework
              ${_caffe_path}/op_proto
              ${_caffe_path}/scripts
  DESTINATION ${_caffe_dest}
)
cann_install(
  TARGET      copy_caffe_op_samples
  FILES       ${_caffe_path}/tbe/CMakeLists.txt
  DIRECTORIES ${_caffe_path}/tbe/impl
              ${_caffe_path}/tbe/op_info_cfg
  DESTINATION ${_caffe_dest}/tbe
)


# install singleop to atc/sample/op/singleop
set(_singleop_path "${CMAKE_CURRENT_SOURCE_DIR}/singleop")
set(_singleop_dest "${SAMPLES_INSTALL_PATH}/singleop")
add_custom_target(copy_singleop_op_samples ALL)
cann_install(
  TARGET      copy_singleop_op_samples
  DIRECTORIES ${_singleop_path}/tbe
  DESTINATION ${_singleop_dest}
)
