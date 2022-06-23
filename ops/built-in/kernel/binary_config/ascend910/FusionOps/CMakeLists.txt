cmake_minimum_required(VERSION 3.5.1)
project(GE_IR[CXX])

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

set(ASCEND_PATH $ENV{ASCEND_RUN_PATH})
set(LOCAL_PATH ./)
set(FWK_INCLUDE_DIR ${ASCEND_PATH}/fwkacllib/include)

set(SRC_LIST
    "transdata_conv2d_transdata.cpp"
)

set(common_compile_options
    -std=c++11
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

######### fwk_ir_build ##########
add_executable(fwk_ir_build
    ${SRC_LIST}
)

target_compile_options(fwk_ir_build PRIVATE
    ${common_compile_options}
)

target_compile_definitions(fwk_ir_build PRIVATE
    ${common_compile_definitions}
)

target_include_directories(fwk_ir_build PRIVATE
    ${LOCAL_PATH}/../../../../op_proto/inc
    ${FWK_INCLUDE_DIR}/graph
    ${FWK_INCLUDE_DIR}/ge
    ${FWK_INCLUDE_DIR}
)

target_link_directories(fwk_ir_build PRIVATE
    ${ASCEND_PATH}/fwkacllib/lib64/stub
)

target_link_libraries(fwk_ir_build PRIVATE
    -Wl,--no-as-needed
    graph
    ge_runner
    fmk_parser
    -Wl,--as-needed
)

######## install ############
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_LIST_DIR})

install(TARGETS fwk_ir_build OPTIONAL
    RUNTIME DESTINATION output
)
