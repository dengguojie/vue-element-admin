project(ops_benchmark)

file(GLOB_RECURSE BENCHMARK_SRCS CONFIGURE_DEPENDS "*.cc"
    ${OPS_DIR}/built-in/tests/common/utils/kernel_run_context_faker.cc)

add_executable(ops_benchmark ${BENCHMARK_SRCS})

target_link_libraries(ops_benchmark PRIVATE $<BUILD_INTERFACE:intf_pub>)

target_include_directories(ops_benchmark PRIVATE
    ${OPS_INCLUDE}
    ${OPS_DIR}/built-in
    ${OPS_DIR}/built-in/op_tiling
    ${OPS_DIR}/built-in/op_proto/inc
    ${FWKACLLIB_INCLUDE}
    ${METADEF_DIR}
    ${METADEF_INCLUDE}
    ${METADEF_INCLUDE}/external
    ${METADEF_INCLUDE}/external/graph
    ${METADEF_INCLUDE}/graph
    ${METADEF_INCLUDE}/exe_graph
    ${GRAPHENGINE_INCLUDE}
    ${GRAPHENGINE_INCLUDE}/external
    ${JSON_INCLUDE}
    ${Protobuf_INCLUDE}
    ${CANN_ROOT}/build/cann/third_party/src/secure_c/include
    ${OPS_DIR}/built-in/tests/common/utils
)

target_link_libraries(ops_benchmark PUBLIC
    benchmark
    benchmark_main
    ascend_protobuf
    graph
    exe_graph
    register
    optiling
    opsproto
)

set_target_properties(ops_benchmark PROPERTIES CXX_STANDARD 11)
target_compile_options(ops_benchmark PRIVATE -O2)

add_custom_command(
    TARGET ops_benchmark POST_BUILD
    COMMAND export LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ASCEND_ATC_DIR} && ./ops_benchmark
    COMMENT "Run ops benchmark test")