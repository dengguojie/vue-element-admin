LOCAL_PATH := $(call my-dir)

# common definitions for plugin compilation
## cloud only supports python3.6
ifeq ($(product),cloud)
    PYTHON_FLAGS := /usr/local/python3.6/include/python3.6m
    PYTHON_LDFLAGS := -L/usr/local/python3.6/lib -lpython3.6m
else
    ifeq ($(rc),arm-server)
        PYTHON_FLAGS  += -I$(ROOTDIR)/../third_party/python_aarch64/Python-3.5.2/Include \
                          -I$(ROOTDIR)/../third_party/python_aarch64/Python-3.5.2/aarch64-linux-gnu/python3.5m
        PYTHON_LDFLAGS += -L$(ROOTDIR)/../third_party/python_aarch64/Python-3.5.2 -lpython3.5m
    else
        PYTHON_FLAGS := /usr/include/python3.5m
        ifneq ($(filter centos euleros,$(host_os)),)
            PYTHON_LDFLAGS := -L/usr/lib -lpython3.5m
        else
            PYTHON_LDFLAGS := -L/usr/lib/x86_64-linux-gnu -lpython3.5m
        endif
    endif
endif
# omg dep
ifeq ($(product),cloud)
LIBGE := libge_train
else
LIBGE := libge
endif

# includes path for plugin compilation
PLUGIN_C_INCLUDES := \
        $(LOCAL_PATH)/../../framework/domi \
        $(LOCAL_PATH)/../../inc \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/../../inc/cce \
        $(LOCAL_PATH)/../../inc/graph \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../third_party/json/include \
        $(LOCAL_PATH)/../../libc_sec/include \
        $(LOCAL_PATH)/framework/tf_plugin/util \
        $(PYTHON_FLAGS)
# ldflags for plugin compilation
PLUGIN_LDFLAGS := \
        $(PYTHON_LDFLAGS)
# shared libs for plugin compilation
PLUGIN_SHARED_LIBS := \
        $(LIBGE) \
        libte_fusion \
        libgraph \
        lib_caffe_parser \
        libc_sec \
        libprotobuf

OPS_PKG_SHARED_LIBS := \
        libops_custom_plugin

#########################################
# tbe plugin for all
include $(CLEAR_VARS)

LOCAL_MODULE := libops_custom_plugin

ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
endif

LOCAL_SRC_FILES := \
        framework/tf_plugin/reduce_prod_plugin.cpp

LOCAL_C_INCLUDES := \
        $(PLUGIN_C_INCLUDES)

LOCAL_LDFLAGS := \
        $(PLUGIN_LDFLAGS)

LOCAL_SHARED_LIBRARIES := \
        $(PLUGIN_SHARED_LIBS)

include $(BUILD_HOST_SHARED_LIBRARY)
#########################################

##############make ops run###############

include $(CLEAR_VARS)

#compile ops run
.PHONY: ops_custom_version_run.run
ops_custom_version_run.run: $(OPS_PKG_SHARED_LIBS)
	@mkdir -p $(dir $@)
	@echo "=========================="
	@echo $(dir $@)
	@rm -rf .$(TOP_DIR)/ops/custom/makepkg
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/fusion_rules/custom/
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/op_proto/custom/
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/framework/custom/tensorflow/
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/op_impl/custom/tbe/
	@mkdir -p .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/op_impl/custom/ai_core/tbe/config/
	@cp .$(TOP_DIR)/out/$(product)/host/libops_custom_plugin.so .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/framework/custom/tensorflow/
	@cp -r .$(TOP_DIR)/ops/custom/tbe/impl .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/op_impl/custom/tbe/
	@cp -r .$(TOP_DIR)/ops/custom/fusion_rules/* .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/fusion_rules/custom/
	@python ${TOP_DIR}/ops/custom/cmake/util/parse_ini_to_json.py `ls ${TOP_DIR}/ops/custom/tbe/op_info_cfg/ai_core/*.ini`
	@cp -f ./tbe_ops_info.json .$(TOP_DIR)/ops/custom/makepkg/ops_custom_pkg/op_impl/custom/ai_core/tbe/config/aic_ops_info.json
	@rm -f ./tbe_ops_info.json
	@cp .$(TOP_DIR)/ops/custom/scripts/install.sh .$(TOP_DIR)/ops/custom/makepkg/run_customops_install.sh
	@.$(TOP_DIR)/release/pkgtools/makeself/makeself.sh --gzip --complevel 4 --nomd5 --sha256 .$(TOP_DIR)/ops/custom/makepkg ops_custom_version_run.run "version:1.0" ./run_customops_install.sh
	@mv .$(TOP_DIR)/ops_custom_version_run.run out/$(product)/host/obj/
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

