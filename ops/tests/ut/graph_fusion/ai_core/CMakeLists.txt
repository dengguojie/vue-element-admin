# graph fusion ai core test
enable_testing()

include_directories(${REPO_ROOT}/libc_sec/include)
include_directories(${REPO_ROOT}/third_party/protobuf/include)

# TODO: used ge/ge_api_error_codes.h, why this is not in atc, should be done after open fusion pass
include_directories(${REPO_ROOT}/inc/external)

# TODO: delete it after source code optimize
include_directories(${REPO_ROOT}/inc/fusion_engine)
include_directories(${REPO_ROOT}/inc)
include_directories(${REPO_ROOT}/inc/register)
include_directories(${REPO_ROOT}/inc/framework)

# for task.pb.h
include_directories(${OP_TEST_ROOT})
# header file in ops/built-in/
include_directories(${OPS_SOURCE_CODE_ROOT}/op_proto/inc)
include_directories(${OPS_SOURCE_CODE_ROOT}/fusion_pass/common)
include_directories(${OPS_SOURCE_CODE_ROOT}/../common/inc)

aux_source_directory(${OPS_SOURCE_CODE_ROOT}/fusion_pass/graph_fusion/ai_core PASS_SRC_FILES)
aux_source_directory(${OPS_SOURCE_CODE_ROOT}/fusion_pass/graph_fusion/ai_core/const2attr PASS_SRC_FILES)
aux_source_directory(${OPS_SOURCE_CODE_ROOT}/fusion_pass/common PASS_SRC_FILES)
message(STATUS "OPS_SOURCE_CODE_ROOT:${OPS_SOURCE_CODE_ROOT}")
# build fusion pass ai core source share library
add_library(fusion_pass_ai_core_source SHARED ${PASS_SRC_FILES})


include_directories(${OP_TEST_ROOT}/common/src/inc)

# scan test case files
file(GLOB FUSION_PASS_AIC_TC_FILES *.cpp)

add_executable(graph_fusion_test ${FUSION_PASS_AIC_TC_FILES})

if ("x${NEED_PRINT_LOG}" STREQUAL "x")
    target_link_libraries(graph_fusion_test
            fusion_pass_ai_core_source gtest op_proto_source c_sec slog pthread graph register error_manager op_test_frame)
else ()
    target_link_libraries(graph_fusion_test
            fusion_pass_ai_core_source gtest op_proto_source slog_ops_stub c_sec slog pthread graph register error_manager op_test_frame)
endif ()

