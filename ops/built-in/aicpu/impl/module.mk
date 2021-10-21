LOCAL_PATH := $(call my-dir)

local_normalized_kernels := kernels/normalized/cast_kernels.cc \
                            kernels/normalized/ceil.cc \
                            kernels/normalized/concatv2.cc \
                            kernels/normalized/expanddims.cc \
                            kernels/normalized/get_dynamic_dims.cc \
                            kernels/normalized/identity.cc \
                            kernels/normalized/less.cc \
                            kernels/normalized/logging.cc \
                            kernels/normalized/masked_select.cc \
                            kernels/normalized/masked_select_grad.cc \
                            kernels/normalized/reshape.cc \
                            kernels/normalized/realdiv.cc \
                            kernels/normalized/round.cc \
                            kernels/normalized/sparse_to_dense_kernels.cc \
                            kernels/normalized/spatial_transformer.cc \
                            kernels/normalized/strided_slice.cc \
                            kernels/normalized/top_k.cc \
                            kernels/normalized/top_k_v2_d.cc \
                            utils/bcast.cc \
                            utils/broadcast_iterator.cc \
                            utils/eigen_tensor.cc \
                            utils/kernel_util.cc \
                            utils/sparse_group.cc \
                            utils/sparse_tensor.cc \

local_host_kernels := kernels/host/add_kernel.cc \
                      kernels/host/mul_kernel.cc \
                      kernels/host/random_uniform_kernel.cc \

local_kernels_inc_path := $(LOCAL_PATH) \
                          $(LOCAL_PATH)/utils \
                          $(TOPDIR)inc/aicpu/cpu_kernels \
                          $(TOPDIR)inc/external/aicpu \
                          ${TOPDIR}third_party/eigen/src/eigen-3.3.7 \
                          $(TOPDIR)inc \
                          $(TOPDIR)libc_sec/include \

# built libcpu_kernels.so for device
include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels
LOCAL_SRC_FILES := $(local_normalized_kernels)
LOCAL_C_INCLUDES := $(local_kernels_inc_path)
LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -Dgoogle=ascend_private -O2 -ftrapv -std=c++14 -fvisibility-inlines-hidden -fvisibility=hidden -DEIGEN_NO_DEBUG -DNDEBUG -DEIGEN_HAS_CXX11_MATH -DEIGEN_OS_GNULINUX
LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl
LOCAL_LDFLAGS += -Wl,-Bsymbolic -Wl,--exclude-libs=libascend_protobuf.a
LOCAL_WHOLE_STATIC_LIBRARIES := libcpu_kernels_context
LOCAL_STATIC_LIBRARIES += libascend_protobuf
LOCAL_SHARED_LIBRARIES := libslog libc_sec

ifeq ($(device_os), android)
    LOCAL_LDLIBS += -llog
endif
include $(BUILD_SHARED_LIBRARY)

# built libcpu_kernels_v1.0.1.so for device
include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels_v1.0.1
LOCAL_SRC_FILES := $(local_normalized_kernels)
LOCAL_C_INCLUDES := $(local_kernels_inc_path)
LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -Dgoogle=ascend_private -O2 -ftrapv -std=c++14 -fvisibility-inlines-hidden -fvisibility=hidden -DEIGEN_NO_DEBUG -DNDEBUG -DEIGEN_HAS_CXX11_MATH -DEIGEN_OS_GNULINUX
LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl
LOCAL_LDFLAGS += -Wl,-Bsymbolic -Wl,--exclude-libs=libascend_protobuf.a
LOCAL_WHOLE_STATIC_LIBRARIES := libcpu_kernels_context
LOCAL_STATIC_LIBRARIES += libascend_protobuf
LOCAL_SHARED_LIBRARIES := libslog libc_sec

ifeq ($(device_os), android)
    LOCAL_LDLIBS += -llog
endif
include $(BUILD_SHARED_LIBRARY)

# built libcpu_kernels.so for host
include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels
LOCAL_SRC_FILES := $(local_normalized_kernels) \
                   $(local_host_kernels)
LOCAL_C_INCLUDES := $(local_kernels_inc_path)
LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv
LOCAL_CFLAGS += -fvisibility-inlines-hidden
LOCAL_CFLAGS += -fvisibility=hidden
LOCAL_LDFLAGS += -Wl,-z,relro,-z,now -s -ldl -shared
LOCAL_LDFLAGS += -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
LOCAL_SHARED_LIBRARIES := libslog libc_sec libcpu_kernels_context

include $(BUILD_HOST_SHARED_LIBRARY)

# built libcpu_kernels.a for host
include $(CLEAR_VARS)

LOCAL_MODULE := libcpu_kernels
LOCAL_SRC_FILES := $(local_normalized_kernels) \
                   $(local_host_kernels)
LOCAL_C_INCLUDES := $(local_kernels_inc_path)
LOCAL_CFLAGS += -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 -ftrapv -DVISIBILITY -std=c++11

include $(BUILD_HOST_STATIC_LIBRARY)
