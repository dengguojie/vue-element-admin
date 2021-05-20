option(PYTHON_CMD "python command")
if (x"${PYTHON_CMD}" STREQUAL x"OFF")
    set(PYTHON_CMD python3.7)
endif ()
set(CMAKE_CXX_FLAGS "-std=c++11")
aux_source_directory(src/model_run_tool ST_TOOL_SRC)

include_directories(
        ./src/model_run_tool
        ../../../inc
)

#add_library(model_run_tool SHARED ${ST_TOOL_SRC})

include_directories(
        src/inc
        ${OP_TEST_ROOT}
        ${REPO_ROOT}/inc
        ${REPO_ROOT}/inc/external
        ${REPO_ROOT}/inc/fusion_engine
        ${REPO_ROOT}/inc/framework
        ${REPO_ROOT}/third_party/protobuf/include
        ${REPO_ROOT}/open_source/protobuf/include
)
aux_source_directory(src/fusion_pass_utils OP_TEST_SRC)
add_library(op_test_frame SHARED ${OP_TEST_SRC})
message(STATUS "CMAKE_CURRENT_BINARY_DIR --------:${CMAKE_CURRENT_BINARY_DIR}")

set(op_test_frame_root ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "op_test_frame_root: ${op_test_frame_root}")
add_custom_target(op_test_frame.whl ALL DEPENDS model_run_tool)
add_custom_command(TARGET op_test_frame.whl
        PRE_BUILD
        COMMAND echo ${op_test_frame_root}
        COMMAND rm -rf ${op_test_frame_root}/python/op_test_frame/libs
        COMMAND mkdir ${op_test_frame_root}/python/op_test_frame/libs
        COMMAND cp libmodel_run_tool.so ${op_test_frame_root}/python/op_test_frame/libs
        COMMAND ${op_test_frame_root}/python/build_whl.sh ${PYTHON_CMD} ${op_test_frame_root}/python
        COMMAND cp ${op_test_frame_root}/python/dist/* ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND rm -rf ${op_test_frame_root}/python/build
        COMMAND rm -rf ${op_test_frame_root}/python/op_test_frame.egg-info
        COMMAND rm -rf ${op_test_frame_root}/python/dist
        )

