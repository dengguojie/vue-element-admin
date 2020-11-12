set(CMAKE_VERBOSE_MAKEFILE on)
include_directories(${OPS_SOURCE_CODE_ROOT}/../)
include_directories(${OPS_SOURCE_CODE_ROOT}/../common/inc)
include_directories(${OPS_SOURCE_CODE_ROOT}/op_tiling)

include_directories(${REPO_ROOT}/libc_sec/include)
include_directories(${REPO_ROOT}/third_party/json/include)
include_directories(${REPO_ROOT}/third_party/protobuf/include)

# fix inc
include_directories(${INC_FIX_ROOT})


aux_source_directory(${OPS_SOURCE_CODE_ROOT}/op_tiling OP_TILING_SOURCE)
set(OP_TILING_SOURCE ${OP_TILING_SOURCE}
        ${OPS_SOURCE_CODE_ROOT}/fusion_pass/common/fp16_t.cpp
        ${OPS_SOURCE_CODE_ROOT}/op_proto/util/error_util.cpp)

add_library(op_tiling_source SHARED ${OP_TILING_SOURCE})

enable_testing()
set(GTEST_ROOT ${TOP_DIR}/llt/third_party/gtest/googletest)

include_directories(${GTEST_ROOT}/include)

message(STATUS "OPS_SOURCE_CODE_ROOT======:${OPS_SOURCE_CODE_ROOT}")
include_directories(${OPS_SOURCE_CODE_ROOT}/op_tiling)
include_directories(${REPO_ROOT}/common)
include_directories(${INC_FIX_ROOT}/external/graph)
include_directories(${INC_FIX_ROOT}/external)
include_directories(${INC_FIX_ROOT})
# for om.pb.h
include_directories(${OP_TEST_ROOT})

message(STATUS "OP_TEST_ROOT========:${OP_TEST_ROOT}")
file(GLOB PLUGIN_TILING_TC_FILES *.cpp)

add_executable(ops_op_tiling_test ${PLUGIN_TILING_TC_FILES})

target_link_libraries (ops_op_tiling_test gtest c_sec slog error_manager graph register op_tiling_source)
