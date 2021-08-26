#!/bin/bash

PWD_DIR=`pwd`

SRC_DIR=${PWD_DIR}/../msopgen/op_gen/template
DST_DIR=${PWD_DIR}/operator_demo_projects

echo "Start operator demo build.sh!"
##############0.init ##########################
#1. make operator_demo_projects file folder
if [ ! -x "${DST_DIR}" ]; then
    mkdir ${DST_DIR}
else
    rm -rf ${DST_DIR}
    mkdir ${DST_DIR}
fi
#2. copy operator impl/ini/proto/plugin to dest
cp -r ${PWD_DIR}/aicpu_operator_sample ${PWD_DIR}/mindspore_operator_sample ${PWD_DIR}/tbe_operator_sample ${DST_DIR}
#3. create a cmakelist file for tf plugin
TF_PLUGIN_CMAKELIST=${PWD_DIR}/operator_demo_projects/tf_plugin_CMakeLists.txt
if [ ! -f "${TF_PLUGIN_CMAKELIST}" ]; then
    touch "${TF_PLUGIN_CMAKELIST}"
else
    rm -rf ${TF_PLUGIN_CMAKELIST}
    touch "${TF_PLUGIN_CMAKELIST}"
fi
echo "# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
aux_source_directory(. SRCS)
message(STATUS \"SRCS = \${SRCS}\")

if(\"x\${SRCS}\" STREQUAL \"x\")
    add_custom_target(\${TF_PLUGIN_TARGET}
            COMMAND mkdir -p \${TF_PLUGIN_TARGET_OUT_DIR}
            COMMAND echo \"no source to make lib \${TF_PLUGIN_TARGET}.so\")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH \${TF_PLUGIN_TARGET_OUT_DIR})

add_library(\${TF_PLUGIN_TARGET} SHARED \${SRCS})

target_compile_definitions(\${TF_PLUGIN_TARGET} PRIVATE
    google=ascend_private
)

target_link_libraries(\${TF_PLUGIN_TARGET} \${ASCEND_INC}/../lib64/libgraph.so)">>${TF_PLUGIN_CMAKELIST}

##############1. copy aicpu_operator_sample ###################
cp -rf ${SRC_DIR}/op_project_tmpl/cmake ${DST_DIR}/aicpu_operator_sample/cmake
cp -rf ${SRC_DIR}/cpukernel/CMakeLists.txt ${DST_DIR}/aicpu_operator_sample/cpukernel/CMakeLists.txt
cp -rf ${SRC_DIR}/cpukernel/toolchain.cmake ${DST_DIR}/aicpu_operator_sample/cpukernel/toolchain.cmake
cp -rf ${SRC_DIR}/op_project_tmpl/framework/CMakeLists.txt ${DST_DIR}/aicpu_operator_sample/framework/CMakeLists.txt
cp -rf ${SRC_DIR}/op_project_tmpl/op_proto/CMakeLists.txt ${DST_DIR}/aicpu_operator_sample/op_proto/CMakeLists.txt
cp -rf ${SRC_DIR}/op_project_tmpl/CMakeLists.txt ${DST_DIR}/aicpu_operator_sample/CMakeLists.txt
cp -rf ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/aicpu_operator_sample/framework/tf_plugin/CMakeLists.txt

##############2. copy tbe_operator_sample ###################
cp -rf ${SRC_DIR}/op_project_tmpl/cmake ${DST_DIR}/tbe_operator_sample/cmake
cp -rf ${SRC_DIR}/op_project_tmpl/framework/CMakeLists.txt ${DST_DIR}/tbe_operator_sample/framework/CMakeLists.txt
cp -rf ${SRC_DIR}/op_project_tmpl/op_proto/CMakeLists.txt ${DST_DIR}/tbe_operator_sample/op_proto/CMakeLists.txt
cp -rf ${SRC_DIR}/tbe/CMakeLists.txt ${DST_DIR}/tbe_operator_sample/tbe/CMakeLists.txt
cp -rf ${SRC_DIR}/op_project_tmpl/CMakeLists.txt ${DST_DIR}/tbe_operator_sample/CMakeLists.txt
cp -rf ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/tbe_operator_sample/framework/tf_plugin/CMakeLists.txt

##############3. clean ###################
rm -rf ${TF_PLUGIN_CMAKELIST}

echo "End operator sample build.sh!"
