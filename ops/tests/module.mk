LOCAL_PATH := $(call my-dir)
# ############ ops ut ###############
include $(CLEAR_VARS)
LOCAL_CLASSFILE_RULE := ops_python

LOCAL_MODULE := ops_ut_python

TEST_CPP_FILES = \
LOCAL_SRC_FILES := \

LOCAL_C_INCLUDES := \

include $(LOCAL_PATH)/ops_python_ut_rule.mk

############ ops ut proto ###############

LOCAL_TEST_DIRS_PATH := $(LOCAL_PATH)/common/utils_plugin_and_op_proto \
                        $(LOCAL_PATH)/stub/plugin_stub.cpp \
                        $(LOCAL_PATH)/ut/ops_test \

LOCAL_TEST_SRC_FILES_SUFFIX := %proto.cpp %proto.cc %tiling.cpp %tiling.cc

LOCAL_TEST_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_TEST_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_TEST_SRC_ALL_FILES := $(LOCAL_TEST_SRC_ALL_FILES:$(LOCAL_TEST_DIRS_PATH)/./%=$(LOCAL_TEST_DIRS_PATH)%)
LOCAL_TEST_SRC_TARGET_FILES := $(filter $(LOCAL_TEST_SRC_FILES_SUFFIX),$(LOCAL_TEST_SRC_ALL_FILES))
LOCAL_TEST_SRC_TARGET_FILES := $(LOCAL_TEST_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_OP_PROTO_FILES_SUFFIX := %.cpp %.cc
LOCAL_OP_PROTO_DIR_PATH := ../../../ops/built-in/op_proto \

LOCAL_OP_PROTO_ALL_FILES := $(foreach src_path,$(LOCAL_OP_PROTO_DIR_PATH), $(shell cd $(LOCAL_PATH) && find "$(src_path)" -type f) )
LOCAL_OP_PROTO_ALL_FILES := $(LOCAL_OP_PROTO_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_OP_PROTO_ALL_TARGET_FILES := $(filter $(LOCAL_OP_PROTO_FILES_SUFFIX),$(LOCAL_OP_PROTO_ALL_FILES))
LOCAL_OP_PROTO_ALL_TARGET_FILES := $(LOCAL_OP_PROTO_ALL_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_OP_TILING_FILES_SUFFIX := %.cpp %.cc
LOCAL_OP_TILING_DIR_PATH := ../../../ops/built-in/op_tiling \

LOCAL_OP_TILING_ALL_FILES := $(foreach src_path,$(LOCAL_OP_TILING_DIR_PATH), $(shell cd $(LOCAL_PATH) && find "$(src_path)" -type f) )
LOCAL_OP_TILING_ALL_FILES := $(LOCAL_OP_TILING_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_OP_TILING_ALL_TARGET_FILES := $(filter $(LOCAL_OP_TILING_FILES_SUFFIX),$(LOCAL_OP_TILING_ALL_FILES))
LOCAL_OP_TILING_ALL_TARGET_FILES := $(LOCAL_OP_TILING_ALL_TARGET_FILES:$(LOCAL_PATH)/%=%)
LOCAL_OP_TILING_ALL_TARGET_FILES += ../../../ops/built-in/fusion_pass/common/fp16_t.cpp


proto_test_utils_files := common/utils_plugin_and_op_proto/op_proto_test_util.cpp

common_proto_files := proto/insert_op.proto

common_shared_libraries := \
    libc_sec \
    libprotobuf \
    libslog_ops_stub \
    libgraph \
    libregister \
    libslog  \
    liberror_manager \

common_c_includes := \
    $(TOPDIR)inc \
    $(TOPDIR)common \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)llt/ops/llt_new/common/utils_plugin_and_op_proto \
    $(TOPDIR)ops/built-in/op_proto/inc \
    $(TOPDIR)ops/built-in/op_proto/util \
    $(TOPDIR)ops/common/inc \
    $(TOPDIR)ops \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)third_party/protobuf/include \
    $(TOPDIR)third_party/json/include \

LOCAL_SRC_FILES := $(common_proto_files)
LOCAL_SRC_FILES += $(LOCAL_TEST_SRC_TARGET_FILES)
LOCAL_SRC_FILES += $(proto_test_utils_files)
LOCAL_SRC_FILES += ut/test_plugin_and_op_proto_main.cpp
LOCAL_SRC_FILES += $(LOCAL_OP_PROTO_ALL_TARGET_FILES)
LOCAL_SRC_FILES += $(LOCAL_OP_TILING_ALL_TARGET_FILES)

$(info "PROTO_LOCAL_SRC_FILES=$(LOCAL_SRC_FILES)")

######### LOCAT VARIABLE  #########
LOCAL_CLASSFILE_RULE := ops_cpp
LOCAL_MODULE := ops_cpp_proto_utest
LOCAL_C_INCLUDES := $(common_c_includes)
LOCAL_SHARED_LIBRARIES := $(common_shared_libraries)
LOCAL_CFLAGS += -DFWK_USE_SYSLOG=1 -DCFG_BUILD_DEBUG -DREUSE_MEMORY=1 -O0 -g3

include $(BUILD_UT_TEST)

############ fusion pass ai_core test ###############
include $(CLEAR_VARS)
common_src_files := \
    proto/insert_op.proto \
    proto/task.proto \

common_c_includes := \
    $(TOPDIR)inc \
    $(TOPDIR)inc/register \
    $(TOPDIR)inc/fusion_engine \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/graph \
    $(TOPDIR)common/graph \
    $(TOPDIR)common \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)third_party/protobuf/include \
    $(TOPDIR)third_party/json/include \
    $(TOPDIR)ops/common/inc \
    $(TOPDIR)ops/built-in/op_proto/inc \
    $(TOPDIR)ops/built-in/op_proto \
    $(TOPDIR)ops/built-in/op_proto/util \
    $(TOPDIR)ops/common/inc \
    $(TOPDIR)ops/ \
    $(TOPDIR)ops/built-in/fusion_pass/ \
    $(TOPDIR)ops/built-in/fusion_pass/common \
    $(TOPDIR)llt/ops/llt_new/stub/fusion_engine \
    $(TOPDIR)llt/ops/llt_new/common/src/inc \
    $(TOPDIR)llt/third_party/googletest/include \

common_shared_libraries := \
    libc_sec \
    libprotobuf \
    libslog_ops_stub \
    libgraph \
    liberror_manager \
    libregister \
    libslog \

#### auto add test files ####
LOCAL_TEST_DIRS_PATH := $(LOCAL_PATH)/common/src/fusion_pass_utils \
                        $(LOCAL_PATH)/ut/graph_fusion/ai_core \

LOCAL_TEST_SRC_FILES_SUFFIX := %.cpp %.cc

LOCAL_TEST_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_TEST_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_TEST_SRC_ALL_FILES := $(LOCAL_TEST_SRC_ALL_FILES:$(LOCAL_TEST_DIRS_PATH)/./%=$(LOCAL_TEST_DIRS_PATH)%)
LOCAL_TEST_SRC_TARGET_FILES := $(filter $(LOCAL_TEST_SRC_FILES_SUFFIX),$(LOCAL_TEST_SRC_ALL_FILES))
LOCAL_TEST_SRC_TARGET_FILES := $(LOCAL_TEST_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

######### auto add ops_plugin files and op_proto files #########
LOCAL_CLASSFILE_RULE := ops_cpp

LOCAL_MODULE := fusion_pass_ai_core_test
LOCAL_PLUGIN_DIRS_PATH := \
                         ../../../ops/built-in/fusion_pass/graph_fusion/ai_core/ \
                         ../../../ops/built-in/fusion_pass/common/ \
                         ../../../ops/built-in/op_proto/

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc
LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell cd $(LOCAL_PATH) && find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)


LOCAL_SRC_FILES := $(common_src_files)
LOCAL_SRC_FILES += $(LOCAL_TEST_SRC_TARGET_FILES)
LOCAL_SRC_FILES += $(LOCAL_PLUGIN_SRC_TARGET_FILES)

######### LOCAT VARIABLE  #########
LOCAL_C_INCLUDES := $(common_c_includes)
LOCAL_SHARED_LIBRARIES := $(common_shared_libraries)
LOCAL_CFLAGS += -DFWK_USE_SYSLOG=1 -DCFG_BUILD_DEBUG -DREUSE_MEMORY=1 -O0 -g3

include $(BUILD_UT_TEST)
