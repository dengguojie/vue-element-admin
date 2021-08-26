
cmake_minimum_required(VERSION 3.14)
project(cpu_kernels_llt)
if ("${DEPEND}" STREQUAL "TRUE")
    set(CMAKE_CXX_STANDARD 11)
    include(ExternalProject)
    include(${CANN_ROOT}/cmake/util/util.cmake)

    include(${CANN_ROOT}/cmake/external/gtest.cmake)
    include(${CANN_ROOT}/cmake/external/secure_c.cmake)
    include(${CANN_ROOT}/cmake/external/eigen.cmake)
    include(${CANN_ROOT}/cmake/external/protobuf.cmake)
    return()
endif()

if ("${ASCEND_CUSTOM_PATH}" STREQUAL "")
    message(WARNING "ASCEND_CUSTOM_PATH was not set, use env var ASCEND_AICPU_PATH instead.")
    if ("$ENV{ASCEND_AICPU_PATH}" STREQUAL "")
        message(FATAL_ERROR "ASCEND_AICPU_PATH was not set, compile failed.")
        return()
    endif()
    set(ASCEND_CUSTOM_PATH $ENV{ASCEND_AICPU_PATH})
endif()

if ("${CANN_ROOT}" STREQUAL "")
    message(WARNING "CANN_ROOT was not set, use env var CANN_ROOT instead.")
    set(CANN_ROOT $ENV{CANN_ROOT})
endif()

if (NOT "${CANN_ROOT}" STREQUAL "" AND NOT "${CANN_DEPEND}" STREQUAL "FALSE")
if (NOT "${IMPL}" STREQUAL "FALSE")
    set(CMAKE_CXX_STANDARD 11)

    set(AICPU_OPP_ENV ${ASCEND_CUSTOM_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
    set(OP_PATH ${PROJECT_PATH}/cpukernel/impl/)
    set(THIRD_PATH ${PROJECT_PATH}/testcases/libs/aicpu_depend/third_party/src)
    set(PROTO_BINARY_DIR ${PROJECT_PATH}/build/aicpu_ut_impl/where)

    set(Protobuf_INCLUDE ${THIRD_PATH}/external_protobuf/src)
    set(EIGEN_INCLUDE ${THIRD_PATH}/eigen)
    set(C_SEC_INCLUDE ${THIRD_PATH}/secure_c/include)
    set(GTEST_INCLUDE ${THIRD_PATH}/external_gtest/googletest/include)

    add_library(gtest SHARED IMPORTED)
    set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${THIRD_PATH}/external_gtest-build/googlemock/gtest/libgtest.so)

    add_library(gtest_main SHARED IMPORTED)
    set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION  ${THIRD_PATH}/external_gtest-build/googlemock/gtest/libgtest_main.so)

    set(Protobuf_PROTOC_EXECUTABLE ${THIRD_PATH}/external_protobuf-build/protoc)
    add_custom_target(protoc ALL DEPENDS ${Protobuf_PROTOC_EXECUTABLE})

    set(CPU_CONTEXT_DIR ${CANN_ROOT}/ops/built-in/aicpu/context)
    set(CPU_KERNELS_DIR ${CANN_ROOT}/ops/built-in/aicpu/impl)

    include(${CANN_ROOT}/cmake/util/util.cmake)

    set(FWKACLLIB_INCLUDE ${CANN_ROOT}/third_party/fwkacllib/inc)

    set(_proto_srcs
      ${CANN_ROOT}/ops/built-in/tests/ut/aicpu_test/proto/cpu_tensor.proto
      ${CANN_ROOT}/ops/built-in/tests/ut/aicpu_test/proto/cpu_attr.proto
      ${CANN_ROOT}/ops/built-in/tests/ut/aicpu_test/proto/cpu_tensor_shape.proto
      ${CANN_ROOT}/ops/built-in/tests/ut/aicpu_test/proto/cpu_node_def.proto
    )

    # the output path of generated files
    set(_proto_include "${PROTO_BINARY_DIR}/opp")
    protobuf_generate(opp _proto_cc _proto_h ${_proto_srcs})

    set(_cpu_context
      ${CPU_CONTEXT_DIR}/cpu_proto/node_def.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/node_def_impl.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/tensor.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/tensor_impl.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/tensor_shape.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/tensor_shape_impl.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/attr_value.cc
      ${CPU_CONTEXT_DIR}/cpu_proto/attr_value_impl.cc
      ${CPU_CONTEXT_DIR}/common/device.cc
      ${CPU_CONTEXT_DIR}/common/context.cc
      ${CPU_CONTEXT_DIR}/common/device_cpu_kernel.cc
      ${CPU_CONTEXT_DIR}/common/async_cpu_kernel.cc
      ${CPU_CONTEXT_DIR}/common/async_event_util.cc
      ${CPU_CONTEXT_DIR}/common/cpu_kernel_register.cc
      ${CPU_CONTEXT_DIR}/common/cpu_kernel_utils.cc
      ${CPU_CONTEXT_DIR}/common/host_sharder.cc
      ${CPU_CONTEXT_DIR}/common/device_sharder.cc
      ${CPU_CONTEXT_DIR}/common/eigen_threadpool.cc
      ${CPU_CONTEXT_DIR}/common/cpu_kernel_cache.cc
      ${CPU_CONTEXT_DIR}/stub/aicpu_sharder.cc
    )
    if(EXISTS "${OP_PATH}/where_kernels.cc")
        set(OP_IMPL_FILE ${OP_PATH}/where_kernels.cc)
    elseif(EXISTS "${OP_PATH}/where_kernel.cc")
        message("where_kernels.cc was not found, where_kernel.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/where_kernel.cc)
    elseif(EXISTS "${OP_PATH}/where.cc")
        message("where_kernels.cc and where_kernel.cc were not found, "
                "where.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/where.cc)
    else()
        message("where_kernels.cc, where.cc and where_kernel.cc "
                "were not found, will found the impl file whose name matches where_**.cc")
        file(GLOB OP_IMPL_FILE ${OP_PATH}/where_**.cc)
    endif()
    set(_cpu_kernels_src
      ${OP_IMPL_FILE}
      ${CPU_KERNELS_DIR}/utils/bcast.cc
      ${CPU_KERNELS_DIR}/utils/eigen_tensor.cc
      ${CPU_KERNELS_DIR}/utils/kernel_util.cc
      ${CPU_KERNELS_DIR}/utils/sparse_group.cc
      ${CPU_KERNELS_DIR}/utils/sparse_tensor.cc
    )

    set(cpu_kernels_llt_src
      ${_cpu_context}
      ${_cpu_kernels_src}
      ${_proto_cc}
      ${_proto_h}
    )

    add_library(cpu_kernels_llt STATIC
      ${cpu_kernels_llt_src}
    )

    link_directories(${ASCEND_CUSTOM_PATH}/atc/lib64/)

    set(cpu_kernels_llt_include
      ${CPU_KERNELS_DIR}
      ${CPU_KERNELS_DIR}/utils
      ${CPU_CONTEXT_DIR}/inc
      ${CPU_CONTEXT_DIR}/common
      ${CPU_CONTEXT_DIR}/stub
      ${CPU_CONTEXT_DIR}/cpu_proto
      ${AICPU_OPP_ENV}/inc
      ${_proto_include}
      ${Protobuf_INCLUDE}
      ${EIGEN_INCLUDE}
      ${FWKACLLIB_INCLUDE}
      ${FWKACLLIB_INCLUDE}/aicpu/common
      ${OP_PATH}
      ${C_SEC_INCLUDE}
    )

    target_include_directories(cpu_kernels_llt PUBLIC
      ${cpu_kernels_llt_include}
    )

    target_compile_options(cpu_kernels_llt PUBLIC
      -D_GLIBCXX_USE_CXX11_ABI=0
      -O0
      -Dgoogle=ascend_private
      -DRUN_TEST
    )

    target_link_libraries(cpu_kernels_llt
      PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
      PUBLIC
        ascend_protobuf
        c_sec
        -ldl
    )

    file(GLOB TEST_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cpp)
    file(GLOB TEST_FILE_CC ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cc)
    set(_cpu_kernels_ut_files
      ${CANN_ROOT}/ops/built-in/tests/utils/node_def_builder.cpp
      ${CANN_ROOT}/ops/built-in/tests/utils/aicpu_test_utils.cpp
      ${CANN_ROOT}/ops/built-in/tests/utils/aicpu_read_file.cpp
      ${TEST_FILE_CPP}
      ${TEST_FILE_CC}
    )

    add_executable(cpu_kernels_ut
      ${_cpu_kernels_ut_files}
    )

    target_include_directories(cpu_kernels_ut PRIVATE
      ${GTEST_INCLUDE}
      ${CANN_ROOT}/ops/built-in/tests/utils/
    )

    target_link_libraries(cpu_kernels_ut PRIVATE
      -Wl,--whole-archive
      cpu_kernels_llt
      -Wl,--no-whole-archive
    )

    target_compile_definitions(cpu_kernels_ut PUBLIC
      D_GLIBCXX_USE_CXX11_ABI=0
      google=ascend_private
      RUN_TEST
    )
    set(AICPU_UTEST "${PROJECT_PATH}/out/bin/aicpu_where_ut_test")
    set_target_properties(cpu_kernels_ut PROPERTIES OUTPUT_NAME "aicpu_where_ut" RUNTIME_OUTPUT_DIRECTORY  ${AICPU_UTEST})

    # llt
    add_library(intf_llt_pub INTERFACE)

    target_include_directories(intf_llt_pub INTERFACE
      ${GTEST_INCLUDE}
    )

    target_compile_definitions(intf_llt_pub INTERFACE
      _GLIBCXX_USE_CXX11_ABI=0
      CFG_BUILD_DEBUG
    )

    target_compile_options(intf_llt_pub INTERFACE
      -g
      --coverage
      -fprofile-arcs
      -ftest-coverage
      -w
      $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
      $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan>
      -fPIC
    )

    set_target_properties(intf_llt_pub
        PROPERTIES
        INTERFACE_LINK_OPTIONS "-fprofile-arcs -ftest-coverage $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -static-libasan -fsanitize=undefined  -static-libubsan>"
    )

    target_link_libraries(intf_llt_pub INTERFACE
      gtest
      gtest_main
      gcov
      pthread
    )
endif()

if (NOT "${PROTO}" STREQUAL "FALSE")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
    set(GTEST_DIR ${PROJECT_PATH}/testcases/libs/gtest)
    set(ATC_DIR ${ASCEND_CUSTOM_PATH}/atc)
    set(OP_PROTO_SRC_DIR ${PROJECT_PATH}/op_proto)

    message(STATUS "ATC_DIR=${ATC_DIR}")
    include(ExternalProject)
    include(${CANN_ROOT}/cmake/util/util.cmake)
    enable_testing()

    add_custom_target(third_proto)
    include(${CANN_ROOT}/cmake/external/secure_c.cmake)
    add_dependencies(third_proto c_sec)
    include_directories(${CANN_ROOT}/third_party/fwkacllib/inc)
    include_directories(${CANN_ROOT}/third_party/metadef/inc/external)
    include_directories(${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc/)
    include_directories(${CANN_ROOT}/third_party/metadef/third_party/graphengine/inc/external/)
    include_directories(${CANN_ROOT}/third_party/metadef/)
    include_directories(${CANN_ROOT}/third_party/metadef/inc)
    include_directories(${CANN_ROOT}/ops/built-in/op_proto/util)
    include_directories(${CANN_ROOT}/ops/built-in/op_proto/inc)
    include_directories(${CANN_ROOT}/ops/common/inc/)
    include_directories(${CANN_ROOT}/third_party/metadef/inc/external/graph)

    include_directories(
            "${GTEST_DIR}/include"
            "${ATC_DIR}/include"
            "${OP_PROTO_SRC_DIR}"
            )

    aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
    file(GLOB OP_PROTO_TEST_FILES_CPP **proto.cpp)
    file(GLOB OP_PROTO_TEST_FILES_CC **proto.cc)
    file(GLOB UTIL_CC ${CANN_ROOT}/ops/built-in/op_proto/util/**.cc)

    link_directories(
            "${ATC_DIR}/lib64"
            "${GTEST_DIR}"
    )

    set(CUSTOM_OBJECT_NAME "where_proto_test")
    add_executable(${CUSTOM_OBJECT_NAME}
            ${PROJECT_PATH}/testcases/ut/aicpu_test/test_main.cc
            ${OP_PROTO_SOURCE_SRCS}
            ${OP_PROTO_TEST_FILES_CPP}
            ${OP_PROTO_TEST_FILES_CC}
            ${UTIL_CC}
            ${CANN_ROOT}/ops/built-in/tests/stub/slog/src/slog.cpp)
    add_dependencies(${CUSTOM_OBJECT_NAME} third_proto)
    target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec alog pthread error_manager graph register)

endif()
else()
if (NOT "${IMPL}" STREQUAL "FALSE")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    set(CMAKE_CXX_STANDARD 11)
    set(AICPU_OPP_ENV ${ASCEND_CUSTOM_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel)
    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
    set(OP_PATH ${PROJECT_PATH}/cpukernel/impl/)

    add_library(gtest SHARED IMPORTED)
    set_target_properties(gtest PROPERTIES IMPORTED_LOCATION ${PROJECT_PATH}/testcases/libs/gtest/libgtest.a)

    add_library(gtest_main SHARED IMPORTED)
    set_target_properties(gtest_main PROPERTIES IMPORTED_LOCATION ${PROJECT_PATH}/testcases/libs/gtest/libgtest_main.a)

	set(GTEST_INCLUDE  ${PROJECT_PATH}/testcases/libs/gtest/include)

    if(EXISTS "${OP_PATH}/where_kernels.cc")
        set(OP_IMPL_FILE ${OP_PATH}/where_kernels.cc)
    elseif(EXISTS "${OP_PATH}/where_kernel.cc")
        message("where_kernels.cc was not found, where_kernel.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/where_kernel.cc)
    elseif(EXISTS "${OP_PATH}/where.cc")
        message("where_kernels.cc and where_kernel.cc were not found, "
                "where.cc was found as the impl file")
        set(OP_IMPL_FILE ${OP_PATH}/where.cc)
    else()
        message("where_kernels.cc, where.cc and where_kernel.cc "
                "were not found, will found the impl file whose name matches where_**.cc")
        file(GLOB OP_IMPL_FILE ${OP_PATH}/where_**.cc)
    endif()

    file(GLOB TEST_FILE_CPP ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cpp)
    file(GLOB TEST_FILE_CC ${CMAKE_CURRENT_SOURCE_DIR}/**_impl.cc)
    set(_cpu_kernels_llt_files
      ${TEST_FILE_CPP}
      ${TEST_FILE_CC}
      ${OP_IMPL_FILE}
    )

    link_directories(${AICPU_OPP_ENV}/lib/arch64
                     ${AICPU_OPP_ENV}/lib/x86
                     ${ASCEND_CUSTOM_PATH}/atc/lib64)

    add_executable(cpu_kernels_llt
      ${_cpu_kernels_llt_files}
    )

    target_include_directories(cpu_kernels_llt PRIVATE
      ${GTEST_INCLUDE}
      ${ASCEND_CUSTOM_PATH}/opp/op_impl/built-in/aicpu/aicpu_kernel/inc/
      ${OP_PATH}
    )

    target_link_libraries(cpu_kernels_llt
      gtest
      gtest_main
      gcov
      pthread
      cpu_kernels_context
      ascend_protobuf
      alog
      -ldl
    )

    target_compile_options(cpu_kernels_llt PUBLIC
      -g
      -O0
      --coverage
      -fprofile-arcs
      -ftest-coverage
      -w
      -Dgoogle=ascend_private
      $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
      $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fno-omit-frame-pointer -static-libasan -fsanitize=undefined -static-libubsan>
      -fPIC
    )

    set(AICPU_UTEST "${PROJECT_PATH}/out/bin/aicpu_where_ut_test")
    set_target_properties(cpu_kernels_llt PROPERTIES OUTPUT_NAME "aicpu_where_ut" RUNTIME_OUTPUT_DIRECTORY  ${AICPU_UTEST})
endif()
if(NOT "${PROTO}" STREQUAL "FALSE")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

    set(PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
    set(GTEST_DIR ${PROJECT_PATH}/testcases/libs/gtest)
    set(ATC_DIR ${ASCEND_CUSTOM_PATH}/atc)
    set(OP_PROTO_SRC_DIR ${PROJECT_PATH}/op_proto)

    message(STATUS "ATC_DIR=${ATC_DIR}")

    enable_testing()

    include_directories(
            "${GTEST_DIR}/include"
            "${ATC_DIR}/include"
            "${OP_PROTO_SRC_DIR}"
            )

    aux_source_directory(${OP_PROTO_SRC_DIR} OP_PROTO_SOURCE_SRCS)
    file(GLOB OP_PROTO_TEST_FILES_CPP **proto.cpp)
    file(GLOB OP_PROTO_TEST_FILES_CC **proto.cc)

    link_directories(
            "${ATC_DIR}/lib64"
            "${GTEST_DIR}"
    )

    set(CUSTOM_OBJECT_NAME "where_proto_test")
    add_executable(${CUSTOM_OBJECT_NAME}
            ${PROJECT_PATH}/testcases/ut/aicpu_test/test_main.cc
            ${OP_PROTO_SOURCE_SRCS}
            ${OP_PROTO_TEST_FILES_CPP}
            ${OP_PROTO_TEST_FILES_CC})

    target_link_libraries(${CUSTOM_OBJECT_NAME} gtest c_sec alog pthread error_manager graph register)
endif()
endif()
