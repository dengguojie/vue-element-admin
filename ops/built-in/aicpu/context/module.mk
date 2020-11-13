LOCAL_PATH := $(call my-dir)

local_context_src_files := cpu_proto/proto/me_attr.proto \
                           cpu_proto/proto/me_node_def.proto \
                           cpu_proto/proto/me_tensor.proto \
                           cpu_proto/proto/me_tensor_shape.proto \
                           cpu_proto/proto/me_types.proto \
                           cpu_proto/node_def.cc \
                           cpu_proto/node_def_impl.cc \
                           cpu_proto/tensor.cc \
                           cpu_proto/tensor_impl.cc \
                           cpu_proto/tensor_shape.cc \
                           cpu_proto/tensor_shape_impl.cc \
                           cpu_proto/attr_value.cc \
                           cpu_proto/attr_value_impl.cc \
                           common/device.cc \
                           common/context.cc \
                           common/device_cpu_kernel.cc \
                           common/cpu_kernel_register.cc \
                           common/cpu_kernel_utils.cc \
                           common/host_sharder.cc \
                           common/device_sharder.cc \
                           common/eigen_threadpool.cc \

local_context_stub_files := stub/aicpu_sharder.cc \

local_context_inc_path := $(LOCAL_PATH) \
                          $(LOCAL_PATH)/common \
                          $(LOCAL_PATH)/cpu_proto \
                          $(TOPDIR)inc \
                          $(TOPDIR)inc/aicpu \
                          $(TOPDIR)inc/aicpu/common \
                          $(TOPDIR)inc/aicpu/cpu_kernels \
                          $(TOPDIR)inc/external/aicpu \
                          $(TOPDIR)libc_sec/include \
                          $(TOPDIR)third_party/protobuf/include \
                          ${TOPDIR}third_party/eigen/src/eigen-3.3.7 \
                          ${TOPDIR}out/${product}

include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels_context

LOCAL_SRC_FILES := $(local_context_src_files)
LOCAL_C_INCLUDES := $(local_context_inc_path)

LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv -std=c++11 -Dgoogle=ascend_private
LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl -shared
LOCAL_SHARED_LIBRARIES := libslog libc_sec libascend_protobuf

ifeq ($(product)$(chip_id), lhisinpuf10)
    LOCAL_SRC_FILES += $(local_context_stub_files)
else
    LOCAL_SHARED_LIBRARIES += libaicpu_sharder
endif

include $(BUILD_SHARED_LIBRARY)


include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels_context

LOCAL_SRC_FILES := $(local_context_src_files) \
                   $(local_context_stub_files)
LOCAL_C_INCLUDES := $(local_context_inc_path)

LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv -DVISIBILITY -std=c++11 -Dgoogle=ascend_private
LOCAL_CFLAGS += -fvisibility-inlines-hidden
LOCAL_CFLAGS += -fvisibility=hidden

LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl -shared
LOCAL_LDFLAGS += -Wl,-Bsymbolic -Wl,--exclude-libs,ALL

LOCAL_SHARED_LIBRARIES := libslog libc_sec libascend_protobuf

include $(BUILD_HOST_SHARED_LIBRARY)


include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels_context

LOCAL_SRC_FILES := $(local_context_src_files) \
                   $(local_context_stub_files)
LOCAL_C_INCLUDES := $(local_context_inc_path)

LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv -DVISIBILITY -std=c++11 -Dgoogle=ascend_private

LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl -shared
LOCAL_UNINSTALLABLE_MODULE := false

include $(BUILD_HOST_STATIC_LIBRARY)
###########################################
include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels_context

LOCAL_SRC_FILES := $(local_context_src_files)
LOCAL_C_INCLUDES := $(local_context_inc_path)

LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv -DVISIBILITY -std=c++11

LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl -shared
LOCAL_UNINSTALLABLE_MODULE := false

include $(BUILD_STATIC_LIBRARY)
