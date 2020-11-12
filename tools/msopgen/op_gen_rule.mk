LOCAL_PATH := $(call my-dir)

INSTALL_MSOPGEN_WHL_IMAGE := $(HOST_OUT_ROOT)/tensor_utils
MSOPGEN_WHL_OUT := toolchain/tensor_utils/msopgen/util/build
MSOPGEN_WHEEL_IMAGE := $(MSOPGEN_WHL_OUT)/op_gen-0.1-py3-none-any.whl

$(MSOPGEN_WHEEL_IMAGE):
ifneq ($(OBB_PRINT_CMD), true)
	rm -rf toolchain/tensor_utils/msopgen/util/build
	mkdir toolchain/tensor_utils/msopgen/util/build
	cd toolchain/tensor_utils/msopgen/util && ./build_whl.sh $(HI_PYTHON)
endif

$(INSTALL_MSOPGEN_WHL_IMAGE): $(MSOPGEN_WHEEL_IMAGE)
ifneq ($(OBB_PRINT_CMD), true)
	mkdir -p $@
	cp -f $< $@
	rm -rf $(MSOPGEN_WHL_OUT)
endif
