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
# copy metadef dependency
CODE_ROOT_DIR=${PWD_DIR}/../../../../../
mkdir -p ${DST_DIR}/AICPU/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/AICPU/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/AICPU/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/register ${DST_DIR}/AICPU/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/AICPU/Tensorflow/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/AICPU/Tensorflow/framework
# copy cann/ops dependency
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/inc ${DST_DIR}/AICPU/Tensorflow/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/AICPU/Tensorflow/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/aicpu/context ${DST_DIR}/AICPU/Tensorflow/cpukernel
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/utils ${DST_DIR}/AICPU/Tensorflow/cpukernel/impl
# copy op_log.h log.h
mkdir -p ${DST_DIR}/AICPU/Tensorflow/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/AICPU/Tensorflow/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/cpukernel/impl/utils/log.h ${DST_DIR}/AICPU/Tensorflow/cpukernel/impl/utils
# copy CMakeLists.txt modified for dependency
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/AICPU/Tensorflow/op_proto
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/cpukernel/CMakeLists.txt ${DST_DIR}/AICPU/Tensorflow/cpukernel
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/framework/tf_plugin/CMakeLists.txt ${DST_DIR}/AICPU/Tensorflow/framework/tf_plugin/CMakeLists.txt
# prepare thirdparty path
mkdir -p ${DST_DIR}/AICPU/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/AICPU/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_kernel.cmake ${DST_DIR}/AICPU/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/eigen.cmake ${DST_DIR}/AICPU/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/protobuf_static.cmake ${DST_DIR}/AICPU/Tensorflow/third_party
#1.2 PyTorch
#cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/AICPU/PyTorch
#cp -r ${SRC_DIR}/cpukernel/* ${DST_DIR}/AICPU//cpukernel
#1.3 Mindspore
#NA

#1.4 Onnx
#1.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/AICPU/Onnx/
cp -r ${SRC_DIR}/cpukernel/* ${DST_DIR}/AICPU/Onnx/cpukernel
# cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/AICPU/Tensorflow/framework/tf_plugin
# copy metadef dependency
mkdir -p ${DST_DIR}/AICPU/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/AICPU/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/AICPU/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/register ${DST_DIR}/AICPU/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/AICPU/Onnx/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/AICPU/Onnx/framework
# copy cann/ops dependency
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/inc ${DST_DIR}/AICPU/Onnx/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/AICPU/Onnx/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/aicpu/context ${DST_DIR}/AICPU/Onnx/cpukernel
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/utils ${DST_DIR}/AICPU/Onnx/cpukernel/impl
# copy op_log.h log.h
mkdir -p ${DST_DIR}/AICPU/Onnx/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/AICPU/Onnx/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/cpukernel/impl/utils/log.h ${DST_DIR}/AICPU/Onnx/cpukernel/impl/utils
# copy CMakeLists.txt modified for dependency
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/AICPU/Onnx/op_proto
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/cpukernel/CMakeLists.txt ${DST_DIR}/AICPU/Onnx/cpukernel
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/framework/onnx_plugin/CMakeLists.txt ${DST_DIR}/AICPU/Onnx/framework/onnx_plugin/CMakeLists.txt
# prepare thirdparty path
mkdir -p ${DST_DIR}/AICPU/Onnx/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/AICPU/Onnx/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_kernel.cmake ${DST_DIR}/AICPU/Onnx/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/eigen.cmake ${DST_DIR}/AICPU/Onnx/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/protobuf_static.cmake ${DST_DIR}/AICPU/Onnx/third_party
# copy ge_onnx.pb.h and ge_onnx.pb.cc
mkdir -p ${DST_DIR}/AICPU/Onnx/proto/onnx
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/ge_onnx.pb.h ${DST_DIR}/AICPU/Onnx/proto/onnx
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/ge_onnx.pb.cc ${DST_DIR}/AICPU/Onnx/proto/onnx


##############2. copy DSL ###################
#2.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/DSL/Tensorflow/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/DSL/Tensorflow/tbe/
cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/DSL/Tensorflow/framework/tf_plugin
# copy metadef dependency
mkdir -p ${DST_DIR}/DSL/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/DSL/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/DSL/Tensorflow/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/DSL/Tensorflow/framework
# copy cann/ops dependency
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/DSL/Tensorflow/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/util ${DST_DIR}/DSL/Tensorflow/tbe/impl/

# copy op_log.h log.h
mkdir -p ${DST_DIR}/DSL/Tensorflow/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/DSL/Tensorflow/log
sed -i 's/#include <utils\/Log.h>/#include <util\/Log.h>/g' ${DST_DIR}/DSL/Tensorflow/log/op_log.h
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/cpukernel/impl/utils/log.h ${DST_DIR}/DSL/Tensorflow/op_proto/util
# copy CMakeLists.txt modified for dependency
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/DSL/Tensorflow/op_proto
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/framework/tf_plugin/CMakeLists.txt ${DST_DIR}/DSL/Tensorflow/framework/tf_plugin/CMakeLists.txt
# prepare thirdparty path
mkdir -p ${DST_DIR}/DSL/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/DSL/Tensorflow/third_party

#2.2 PyTorch
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/DSL/PyTorch/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/DSL/PyTorch/tbe/
# copy metadef dependency.
mkdir -p ${DST_DIR}/DSL/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/DSL/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/DSL/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/DSL/PyTorch/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/DSL/PyTorch/framework
# copy cann/ops dependency.
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/DSL/PyTorch/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/util ${DST_DIR}/DSL/PyTorch/tbe/impl/
# copy op_log.h
mkdir -p ${DST_DIR}/DSL/PyTorch/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/DSL/PyTorch/log
# copy CMakeList.txt modified for dependency.
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/DSL/PyTorch/op_proto
# prepare thirdparty path
mkdir -p ${DST_DIR}/DSL/PyTorch/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/DSL/PyTorch/third_party
#2.3 MindSpore
#NA

#2.4 Onnx
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/DSL/Onnx/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/DSL/Onnx/tbe/
# copy metadef dependency
mkdir -p ${DST_DIR}/DSL/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/DSL/Onnx/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/DSL/Onnx/metadef
# copy CMakeLists.txt modified for dependency
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/framework/onnx_plugin/CMakeLists.txt ${DST_DIR}/DSL/Onnx/framework/onnx_plugin/CMakeLists.txt
# copy cann/ops dependency
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/inc ${DST_DIR}/DSL/Onnx/op_proto
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/DSL/Onnx/op_proto
# copy op_log.h
mkdir -p ${DST_DIR}/DSL/Onnx/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/DSL/Onnx/log
# copy CMakeList.txt modified for dependency.
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/Onnx/op_proto/CMakeLists.txt ${DST_DIR}/DSL/Onnx/op_proto
# copy ge_onnx.pb.h and ge_onnx.pb.cc
mkdir -p ${DST_DIR}/DSL/Onnx/proto/onnx
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/ge_onnx.pb.h ${DST_DIR}/DSL/Onnx/proto/onnx
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/ge_onnx.pb.cc ${DST_DIR}/DSL/Onnx/proto/onnx

##############3. copy TIK ###################
#3.1 Tensorflow
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/TIK/Tensorflow/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/TIK/Tensorflow/tbe
cp ${TF_PLUGIN_CMAKELIST} ${DST_DIR}/TIK/Tensorflow/framework/tf_plugin
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/TIK/Tensorflow/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/TIK/Tensorflow/framework
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/TIK/Tensorflow/op_proto
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/TIK/Tensorflow/op_proto
# prepare thirdparty path
mkdir -p ${DST_DIR}/TIK/Tensorflow/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/TIK/Tensorflow/third_party
# copy metadef dependency.
mkdir -p ${DST_DIR}/TIK/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/TIK/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/TIK/Tensorflow/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/TIK/Tensorflow/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/TIK/Tensorflow/framework
#copy op_log.h
mkdir -p ${DST_DIR}/TIK/Tensorflow/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/TIK/Tensorflow/log
#copy impl/util
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/util ${DST_DIR}/TIK/Tensorflow/tbe/impl
cp -r ${DST_DIR}/DSL/Tensorflow/tbe/impl/conv2d.py ${DST_DIR}/TIK/Tensorflow/tbe/impl
#3.2 PyTorch
cp -r ${SRC_DIR}/op_project_tmpl/* ${DST_DIR}/TIK/PyTorch/
cp -r ${SRC_DIR}/tbe/* ${DST_DIR}/TIK/PyTorch/tbe
# copy metadef dependency.
mkdir -p ${DST_DIR}/TIK/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/graph ${DST_DIR}/TIK/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/inc ${DST_DIR}/TIK/PyTorch/metadef
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/omg ${DST_DIR}/TIK/PyTorch/framework
cp -r ${CODE_ROOT_DIR}/metadef/third_party/graphengine/inc/framework/common ${DST_DIR}/TIK/PyTorch/framework
#copy cann/ops dependency.
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/ops/built-in/op_proto/util ${DST_DIR}/TIK/PyTorch/op_proto
#copy op_log.h
mkdir -p ${DST_DIR}/TIK/PyTorch/log
cp -r ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_log.h ${DST_DIR}/TIK/PyTorch/log
#copy CMakeList.txt modified for dependency.
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/op_proto/CMakeLists.txt ${DST_DIR}/TIK/PyTorch/op_proto
# prepare thirdparty path
mkdir -p ${DST_DIR}/TIK/PyTorch/third_party
cp -rf ${CODE_ROOT_DIR}/asl/ops/cann/tools/custom_operator_sample/dependency_files/secure_c_proto.cmake ${DST_DIR}/TIK/PyTorch/third_party
#3.3 Mindspore
#NA

##############4. clean ###################
rm -rf ${TF_PLUGIN_CMAKELIST}


echo "End operator sample build.sh!"
