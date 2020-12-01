LOCAL_PATH := $(call my-dir)

#compile x86 libslog_stub

include $(CLEAR_VARS)

LOCAL_MODULE := libslog_ops_stub

LOCAL_C_INCLUDES := \
    $(TOPDIR)inc \

LOCAL_SRC_FILES := \
    src/slog.cpp \

include $(BUILD_LLT_SHARED_LIBRARY)
