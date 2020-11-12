LOCAL_SHARED_LIBRARIES :=   libtvm \
							libtvm_runtime \
							libauto_tiling \
							libplatform \
							simulator/Ascend910/lib/lib_pvmodel \
							simulator/Ascend910/lib/libnpu_drv_pvmodel \
							simulator/Ascend910/lib/libruntime_cmodel \
							simulator/Ascend910/lib/libtsch \
							libslog
include $(BUILD_SYSTEM)/binary.mk

ROOT_DIR:=$(PWD)

ifneq ($(shell echo $(strip $(HOST_LLT_X86_TOOLCHAIN_PREFIX))|grep -c '^/'),1)
LLT_GXX_PATH := $(ROOT_DIR)/$(strip $(HOST_LLT_X86_TOOLCHAIN_PREFIX))g++
LLT_GCC_PATH := $(ROOT_DIR)/$(strip $(HOST_LLT_X86_TOOLCHAIN_PREFIX))gcc
else
LLT_GXX_PATH := $(strip $(HOST_LLT_X86_TOOLCHAIN_PREFIX))g++
LLT_GCC_PATH := $(ROOT_DIR)/$(strip $(HOST_LLT_X86_TOOLCHAIN_PREFIX))gcc
endif

test_mode=2
ifneq ($(op_type),)
test_mode=3
endif

$(LOCAL_BUILT_MODULE) : PRIVATE_LOCAL_PATH:= $(LOCAL_PATH)
$(LOCAL_BUILT_MODULE) : PRIVATE_LOCAL_MODULE := $(LOCAL_MODULE)
$(LOCAL_BUILT_MODULE) : $(all_libraries) $(all_objects)
	cp model/model_v100/pv_model/etc/config_pv_aicore_model.toml $(ROOT_DIR)/out/$(product)/llt/ut/obj/lib/simulator/Ascend910/lib/.
	cp model/model_v100/ca_model/etc/common.spec $(ROOT_DIR)/out/$(product)/llt/ut/obj/lib/simulator/Ascend910/lib/.
	rm -rf toolchain/tensor_utils/op_test_frame/build
	mkdir toolchain/tensor_utils/op_test_frame/build
	cd toolchain/tensor_utils/op_test_frame/build && cmake ../ -DPYTHON_CMD=$(HI_PYTHON) -DFOR_LLT=true -DLLT_GCC_PATH=$(LLT_GCC_PATH) -DLLT_GXX_PATH=$(LLT_GXX_PATH) && make
	rm -rf toolchain/tensor_utils/op_test_frame/build
	.$(TOPDIR)/llt/ops/llt_new/common/ci/run_op_ut.sh $(ROOT_DIR) $(ROOT_DIR)/out/$(product) $(test_mode) $(op_type)
