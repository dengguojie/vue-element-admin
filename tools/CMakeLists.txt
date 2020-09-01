option(PYTHON_CMD "python command" "python3.7")
set(CMAKE_CXX_FLAGS "-std=c++11")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
aux_source_directory(src/model_run_tool ST_TOOL_SRC)

include_directories(
        ./src/model_run_tool
        ../../../inc
)

add_library(model_run_tool SHARED ${ST_TOOL_SRC})

if ('x${PYTHON_CMD}' STREQUAL 'xOFF')
    set(PYTHON_CMD python3.7)
endif()

set(op_test_frame_root ${CMAKE_SOURCE_DIR})
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
