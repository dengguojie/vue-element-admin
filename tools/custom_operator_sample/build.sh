#!/bin/bash

PWD_DIR=`pwd`

SRC_DIR=${PWD_DIR}/../msopgen/op_gen/template
DST_DIR=${PWD_DIR}/custom_operator_sample

echo "Start operator sample build.sh!"
##############0.init ##########################
#1. make custom_operator_sample file folder
if [ ! -x "${DST_DIR}" ]; then
    mkdir ${DST_DIR}
else
    rm -rf ${DST_DIR}
    mkdir ${DST_DIR}
fi
#2. copy operator impl/ini/proto/plugin to dest 
cp -r ${PWD_DIR}/AICPU ${PWD_DIR}/DSL ${PWD_DIR}/TIK ${DST_DIR} 
#3. create a cmakelist file for tf plugin
TF_PLUGIN_CMAKELIST=${PWD_DIR}/custom_operator_sample/CMakeLists.txt
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


##############1. copy AICPU ###################
#1.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/AICPU/Tensorflow/
cp -r ${SRC_DIR}/cpukernel/* ${DST_DIR}/AICPU/Tensorflow/cpukernel
cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/AICPU/Tensorflow/framework/tf_plugin
#1.2 PyTorch
#cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/AICPU/PyTorch
#cp -r ${SRC_DIR}/cpukernel/* ${DST_DIR}/AICPU//cpukernel
#1.3 Mindspore
#NA

##############2. copy DSL ###################
#2.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/DSL/Tensorflow/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/DSL/Tensorflow/
cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/DSL/Tensorflow/framework/tf_plugin
#2.2 PyTorch
#cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/DSL/PyTorch

#2.3 MindSpore
#NA


##############3. copy TIK ###################
#3.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/TIK/Tensorflow/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/TIK/Tensorflow/tbe
cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/TIK/Tensorflow/framework/tf_plugin
#3.2 PyTorch
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/TIK/PyTorch/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/TIK/PyTorch/tbe

#3.3 Mindspore
#NA

##############4. clean ###################
rm -rf ${TF_PLUGIN_CMAKELIST}


echo "End operator sample build.sh!"
