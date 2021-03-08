LOCAL_PATH := $(call my-dir)

INSTALL_OP_TEST_FRAME_WHL_IMAGE := $(HOST_OUT_ROOT)/op_test_frame-0.1-py3-none-any.whl
CANN_OP_TEST_FRAME_WHL_OUT := toolchain/tensor_utils/op_test_frame
OP_TEST_FRAME_WHEEL_IMAGE = $(CANN_OP_TEST_FRAME_WHL_OUT)/build/op_test_frame-0.1-py3-none-any.whl

OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME :=
ifeq ($(HOST_COMPILE_ARCH), x86_64)
OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME := op_test_frame-0.1-cp37-cp37m-linux_x86_64.whl
else
OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME := op_test_frame-0.1-cp37-cp37m-linux_aarch64.whl
endif

$(OP_TEST_FRAME_WHEEL_IMAGE):
ifneq ($(OBB_PRINT_CMD), true)
	rm -rf toolchain/tensor_utils/op_test_frame/build
	mkdir toolchain/tensor_utils/op_test_frame/build
	cd toolchain/tensor_utils/op_test_frame/build && cmake ../ -DHI_PYTHON=$(HI_PYTHON) && make
	echo "find wheel packages:" && ls toolchain/tensor_utils/op_test_frame/build
	mv toolchain/tensor_utils/op_test_frame/build/$(OP_TEST_FRAME_WHEEL_PACKAGE_REAL_NAME) $(OP_TEST_FRAME_WHEEL_IMAGE)
endif

$(INSTALL_OP_TEST_FRAME_WHL_IMAGE): $(OP_TEST_FRAME_WHEEL_IMAGE)
ifneq ($(OBB_PRINT_CMD), true)
	$(call copy-file-to-target)
	rm -rf toolchain/tensor_utils/op_test_frame/build
endif
