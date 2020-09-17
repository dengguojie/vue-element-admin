set(CMAKE_VERBOSE_MAKEFILE on)
include_directories(${OPS_SOURCE_CODE_ROOT}/../)
include_directories(${OPS_SOURCE_CODE_ROOT}/../common/inc)
include_directories(${OPS_SOURCE_CODE_ROOT}/op_proto/inc)
include_directories(${OPS_SOURCE_CODE_ROOT}/op_proto/util)

include_directories(${REPO_ROOT}/libc_sec/include)
include_directories(${REPO_ROOT}/third_party/json/include)
include_directories(${REPO_ROOT}/third_party/protobuf/include)

# fix inc
include_directories(${INC_FIX_ROOT})


aux_source_directory(${OPS_SOURCE_CODE_ROOT}/op_proto/util OP_PROTO_SOURCE)
aux_source_directory(${OPS_SOURCE_CODE_ROOT}/op_proto OP_PROTO_SOURCE)
aux_source_directory(${OPS_SOURCE_CODE_ROOT}/op_tiling OP_TILING_SOURCE)
set(OP_TILING_SOURCE ${OP_TILING_SOURCE}
        ${OPS_SOURCE_CODE_ROOT}/fusion_pass/common/fp16_t.cpp
        ${OPS_SOURCE_CODE_ROOT}/op_proto/util/error_util.cpp)

add_library(op_proto_source SHARED ${OP_PROTO_SOURCE})
add_library(op_tiling_source SHARED ${OP_TILING_SOURCE})
#target_link_libraries(op_proto_source graph register)

enable_testing()
set(GTEST_ROOT ${TOP_DIR}/llt/third_party/gtest/googletest)
#set(MOCKCPP_ROOT ${TOP_DIR}/llt/third_party/mockcpp/6.3)

include_directories(${GTEST_ROOT}/include)

message(STATUS "OPS_SOURCE_CODE_ROOT======:${OPS_SOURCE_CODE_ROOT}")
include_directories(${OPS_SOURCE_CODE_ROOT}/op_proto/inc)
include_directories(${OPS_SOURCE_CODE_ROOT}/op_proto)
include_directories(${REPO_ROOT}/framework/domi)
include_directories(${REPO_ROOT}/common)
include_directories(${INC_FIX_ROOT}/framework)
include_directories(${INC_FIX_ROOT}/external/graph)
include_directories(${INC_FIX_ROOT}/external)
include_directories(${INC_FIX_ROOT})
# for om.pb.h
include_directories(${OP_TEST_ROOT})

message(STATUS "OP_TEST_ROOT========:${OP_TEST_ROOT}")
file(GLOB PLUGIN_PROTO_TC_FILES **/*proto.cpp)
file(GLOB PLUGIN_TILING_TC_FILES **/*tiling.cpp)
aux_source_directory(${OP_TEST_ROOT}/common/utils_plugin_and_op_proto PLUGIN_PROTO_TEST_UTIL)

add_executable(ops_proto_test
        ${OP_TEST_ROOT}/ut/test_plugin_and_op_proto_main.cpp
        ${PLUGIN_PROTO_TC_FILES}
        ${PLUGIN_TILING_TC_FILES}
        ${PLUGIN_PROTO_TEST_UTIL})

target_link_libraries (ops_proto_test gtest c_sec slog pthread error_manager graph register op_proto_source slog_ops_stub op_tiling_source)
