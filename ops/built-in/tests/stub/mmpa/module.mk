LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := libmmpa_stub_for_ops

LOCAL_C_INCLUDES := \
    $(TOPDIR)inc \
    $(TOPDIR)inc/external \
    $(TOPDIR)inc/external/graph \
    $(TOPDIR)inc/framework \
    $(TOPDIR)inc/cce \
    $(TOPDIR)libc_sec/include \
    $(TOPDIR)framework/domi \
    $(TOPDIR)framework/domi/common \
    $(TOPDIR)third_party/glog/include \
    $(TOPDIR)third_party/protobuf/include \

LOCAL_SRC_FILES := \
    src/mmpa_stub.cpp \

include $(BUILD_LLT_SHARED_LIBRARY)
