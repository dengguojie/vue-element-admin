# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

include_directories(
        ${REPO_ROOT}/ops/built-in
        ${REPO_ROOT}/ops/common/inc
        ${REPO_ROOT}/ops/built-in/fusion_pass/common
        ${REPO_ROOT}/inc
        ${REPO_ROOT}/inc/register
        ${REPO_ROOT}/inc/fusion_engine
        ${REPO_ROOT}/inc/external
        ${REPO_ROOT}/inc/framework
        ${REPO_ROOT}/third_party/protobuf/include
        ${REPO_ROOT}/libc_sec/include
)

aux_source_directory(. SRCS)

if("x${SRCS}" STREQUAL "x")
    add_custom_target(${AIC_FUSION_PASS_TARGET}
            COMMAND mkdir -p ${AIC_FUSION_PASS_TARGET_OUT_DIR}
            COMMAND echo "no source to make lib${AIC_FUSION_PASS_TARGET}.so")
    return(0)
endif()

set(LIBRARY_OUTPUT_PATH ${AIC_FUSION_PASS_TARGET_OUT_DIR})

add_library(${AIC_FUSION_PASS_TARGET} SHARED ${SRCS})