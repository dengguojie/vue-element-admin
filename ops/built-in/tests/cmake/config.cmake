set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -fprofile-arcs -ftest-coverage -Dgoogle=ascend_private")

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")
# disable abi
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# endswith testcases mean under a ops project not single test project
message(STATUS "CMAKE_SOURCE_DIR---------:${CMAKE_SOURCE_DIR}")
if (IS_DIRECTORY "${CMAKE_SOURCE_DIR}/testcases")
    set(TOP_DIR ${CMAKE_SOURCE_DIR}/../..)
    set(OP_TEST_ROOT ${CMAKE_SOURCE_DIR}/testcases)
    set(REPO_ROOT ${CMAKE_SOURCE_DIR}/../..)
else ()
    set(TOP_DIR ${CMAKE_SOURCE_DIR}/../../..)
    set(OP_TEST_ROOT ${CMAKE_SOURCE_DIR})
    MESSAGE(STATUS "OP_TEST_ROOT: ${OP_TEST_ROOT}")

    # TODO: after all decoupled, all header file is in repo_root/inc/external, this config should be delete
    set(REPO_ROOT ${CMAKE_SOURCE_DIR}/../../..)
endif ()
set(INC_FIX_ROOT ${REPO_ROOT}/inc)
message(STATUS "INC_FIX_ROOT=========: ${INC_FIX_ROOT}")

set(USE_ATC TRUE)

if (USE_ATC)
    set(MIND_STUDIO_HOME "~/.mindstudio")
    MESSAGE(STATUS "custom project, using custom configuration base on atc")
    # [Config Me] change me to you own atc root dir, please not ends with separator
    #set(ASCEND_ATC_ROOT ${MIND_STUDIO_HOME}/huawei/adk/1.75.T5.0.B050/atc)
    set(ASCEND_ATC_ROOT /usr/local/Ascend/compiler)
    set(ASCEND_TOOL_ROOT ${MIND_STUDIO_HOME}/huawei/adk/1.75.T5.0.B050/tools/simulator/lib/Ascend910/)
    set(ASCEND_ATC_LIB ${ASCEND_ATC_ROOT}/lib64)
    set(ASCEND_ATC_INC ${ASCEND_ATC_ROOT}/include)

    # these two variable will used everywhere for the sub project to find .h and .cpp
    set(TOP_INC_ROOT ${ASCEND_ATC_INC})
    if ("${OPS_SOURCE_CODE_ROOT}" STREQUAL "")
        set(OPS_SOURCE_CODE_ROOT ${TOP_DIR}/asl/ops/cann/ops/built-in)
    endif ()

    link_directories(${ASCEND_ATC_LIB}
            ${ASCEND_TOOL_ROOT}
            ${TOP_DIR}/llt/third_party/gtest/googletest/build
            )

    include_directories(${TOP_INC_ROOT})
    include_directories(${CMAKE_SOURCE_DIR}/common/utils_plugin_and_op_proto)
    include_directories(${CMAKE_SOURCE_DIR}/stub/framework/domi/parser)
    include_directories(${REPO_ROOT}/llt/third_party/gtest/googletest/include)
else ()
    if(EXISTS ${REPO_ROOT}/build/prebuilts/gcc)
        set(CMAKE_C_COMPILER "${REPO_ROOT}/build/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/bin/gcc")
        set(CMAKE_CXX_COMPILER "${REPO_ROOT}/build/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/bin/g++")
    else()
        set(CMAKE_C_COMPILER "${REPO_ROOT}/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/bin/gcc")
        set(CMAKE_CXX_COMPILER "${REPO_ROOT}/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/bin/g++")
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=address -static-libasan -fsanitize=undefined")
    MESSAGE(STATUS "built-in project, using built-in configuration base on repo code")
    # [Config Me] change me to you own repo root dir, please not ends with separator
    set(TOP_DIR ${REPO_ROOT})

    # these two variable will used everywhere for the sub project to find .h and .cpp
    set(TOP_INC_ROOT ${TOP_DIR}/inc/external)
    if ("${OPS_SOURCE_CODE_ROOT}" STREQUAL "")
        set(OPS_SOURCE_CODE_ROOT ${TOP_DIR}/asl/ops/cann/ops/built-in)
    endif ()

    link_directories(${TOP_DIR}/llt/third_party/googletest/lib/4.9
            ${TOP_DIR}/llt/third_party/mockcpp/4.9/lib
            ${TOP_DIR}/out/onetrack/llt/ut/obj/lib
            ${TOP_DIR}llt/third_party/gtest/googlemock/lib/.libs/)

    include_directories(${TOP_INC_ROOT})
    include_directories(${REPO_ROOT}/llt/third_party/gtest/googletest/include)
endif ()

MESSAGE(STATUS "OPS_SOURCE_CODE_ROOT: ${OPS_SOURCE_CODE_ROOT}")
MESSAGE(STATUS "TOPS_SOURCE_CODE_ROOT: ${TOPS_SOURCE_CODE_ROOT}")

if (IS_DIRECTORY ${CMAKE_SOURCE_DIR}/testcases)
    execute_process(
            COMMAND sh ${CMAKE_SOURCE_DIR}/testcases/prepare_proto.sh ${CMAKE_SOURCE_DIR}/testcases/
    )
else ()
    execute_process(
            COMMAND sh ${CMAKE_SOURCE_DIR}/prepare_proto.sh ${CMAKE_SOURCE_DIR}/
    )
endif ()
