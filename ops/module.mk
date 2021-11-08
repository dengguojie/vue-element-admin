LOCAL_PATH := $(call my-dir)

# includes path for plugin compilation
PLUGIN_C_INCLUDES := \
        $(LOCAL_PATH)/common/inc/ \
        $(LOCAL_PATH)/../../inc \
        $(LOCAL_PATH)/../../common/ \
        $(LOCAL_PATH)/../../metadef \
        $(LOCAL_PATH)/../../metadef/inc \
        $(LOCAL_PATH)/../../metadef/inc/external \
        $(LOCAL_PATH)/../../metadef/inc/external/graph \
        $(LOCAL_PATH)/../../graphengine/inc \
        $(LOCAL_PATH)/../../graphengine/inc/external \
        $(LOCAL_PATH)/../../graphengine/inc/external/ge \
        $(LOCAL_PATH)/../../graphengine/inc/framework \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/../../inc/external/graph \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../libc_sec/include \
        proto/caffe/caffe.proto \
        proto/onnx/ge_onnx.proto \
        proto/tensorflow/attr_value.proto      \
        proto/tensorflow/function.proto        \
        proto/tensorflow/graph.proto           \
        proto/tensorflow/node_def.proto        \
        proto/tensorflow/op_def.proto          \
        proto/tensorflow/resource_handle.proto \
        proto/tensorflow/tensor.proto          \
        proto/tensorflow/tensor_shape.proto    \
        proto/tensorflow/types.proto           \
        proto/tensorflow/versions.proto        \

# shared libs for plugin compilation
PLUGIN_SHARED_LIBS := \
        libte_fusion \
        libgraph \
        libregister \
        lib_caffe_parser \
        libc_sec \
        libascend_protobuf \
        libslog  \

OPS_PKG_SHARED_LIBS := \
        libops_all_plugin \
        aic-ascend910-ops-info.json \
        aic-ascend920-ops-info.json \
        aic-ascend310-ops-info.json \
        aic-ascend710-ops-info.json \
        aic-ascend610-ops-info.json \
        aic-ascend615-ops-info.json \
        aic-hi3796cv300es-ops-info.json \
        aic-hi3796cv300cs-ops-info.json \
        aic-sd3403-ops-info.json \
        vector_core_tbe_ops_info.json \
        tf_kernel.json \
        npu_supported_ops.json

# tbe plugin for all
include $(CLEAR_VARS)

LOCAL_MODULE := libops_all_plugin

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif

LOCAL_PLUGIN_DIRS_PATH := $(LOCAL_PATH)/built-in/framework/tf_plugin \
                          $(LOCAL_PATH)/built-in/framework/tf_scope_fusion_passes \

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc

LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_SRC_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES)

#$(warning $(LOCAL_SRC_FILES))

LOCAL_C_INCLUDES := \
        $(PLUGIN_C_INCLUDES)

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations

LOCAL_SHARED_LIBRARIES := \
        $(PLUGIN_SHARED_LIBS) \
        liberror_manager

include $(BUILD_HOST_SHARED_LIBRARY)


# aicore fusion pass
include $(CLEAR_VARS)

LOCAL_MODULE := libops_fusion_pass_aicore

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif


LOCAL_PLUGIN_DIRS_PATH := $(LOCAL_PATH)/built-in/fusion_pass/common\
                          $(LOCAL_PATH)/built-in/fusion_pass/graph_fusion/ai_core\
                          $(LOCAL_PATH)/built-in/fusion_pass/buffer_fusion/ub_fusion/ai_core

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc

LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_SRC_FILES := \
        $(LOCAL_PLUGIN_SRC_TARGET_FILES) \
        built-in/op_proto/util/error_util.cc \

$(warning "---------------print start--------------")
$(warning $(LOCAL_PATH))
$(warning $(LOCAL_SRC_FILES))

LOCAL_C_INCLUDES := \
        proto/task.proto \
        $(LOCAL_PATH)/built-in/fusion_pass/common \
        $(LOCAL_PATH)/built-in/fusion_pass \
        $(LOCAL_PATH)/common/inc/ \
        $(LOCAL_PATH)/built-in/op_proto/util \
        $(LOCAL_PATH)/../../third_party/json/include \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../libc_sec/include/ \
        $(LOCAL_PATH)/../../inc/ \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/../../inc/fusion_engine \
        $(LOCAL_PATH)/../../metadef \
        $(LOCAL_PATH)/../../metadef/inc \
        $(LOCAL_PATH)/../../metadef/inc/external \
        $(LOCAL_PATH)/../../metadef/inc/external/graph \
        $(LOCAL_PATH)/../../metadef/inc/common/opskernel \
        $(LOCAL_PATH)/../../metadef/inc/register \
        $(LOCAL_PATH)/../../graphengine/inc \
        $(LOCAL_PATH)/../../graphengine/inc/external \
        $(LOCAL_PATH)/../../graphengine/inc/framework \
        $(LOCAL_PATH)/../../fusion_engine/inc \


LOCAL_LDFLAGS := -ldl

LOCAL_SHARED_LIBRARIES := libslog libc_sec libgraph libregister libplatform liberror_manager libaicore_utils

include $(BUILD_HOST_SHARED_LIBRARY)

# aicore fusion pass
include $(CLEAR_VARS)

LOCAL_MODULE := libops_fusion_pass_vectorcore

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif

LOCAL_PLUGIN_DIRS_PATH := $(LOCAL_PATH)/built-in/fusion_pass/common\
                          $(LOCAL_PATH)/built-in/fusion_pass/graph_fusion/vector_core\
                          $(LOCAL_PATH)/built-in/fusion_pass/buffer_fusion/ub_fusion/vector_core

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc

LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_SRC_FILES := \
        $(LOCAL_PLUGIN_SRC_TARGET_FILES) \
        built-in/op_proto/util/error_util.cc \

$(warning "---------------print start--------------")
$(warning $(LOCAL_PATH))
$(warning $(LOCAL_SRC_FILES))

LOCAL_C_INCLUDES := \
        proto/task.proto \
        $(LOCAL_PATH)/built-in/fusion_pass/common \
        $(LOCAL_PATH)/built-in/fusion_pass/vector_core \
        $(LOCAL_PATH)/common/inc/ \
        $(LOCAL_PATH)/built-in/op_proto/util \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../libc_sec/include/ \
        $(LOCAL_PATH)/../../inc/ \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/../../inc/common/opskernel \
        $(LOCAL_PATH)/../../inc/fusion_engine \
        $(LOCAL_PATH)/../../metadef \
        $(LOCAL_PATH)/../../metadef/inc \
        $(LOCAL_PATH)/../../metadef/inc/external \
        $(LOCAL_PATH)/../../metadef/inc/external/graph \
        $(LOCAL_PATH)/../../metadef/inc/common/opskernel \
        $(LOCAL_PATH)/../../metadef/inc/register \
        $(LOCAL_PATH)/../../graphengine/inc \
        $(LOCAL_PATH)/../../graphengine/inc/external \
        $(LOCAL_PATH)/../../graphengine/inc/framework \


LOCAL_LDFLAGS := -ldl

LOCAL_SHARED_LIBRARIES := libslog libc_sec libgraph libregister libplatform liberror_manager

include $(BUILD_HOST_SHARED_LIBRARY)

# libopsproto.so
include $(CLEAR_VARS)

LOCAL_MODULE := libopsproto

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif
LOCAL_SRC_FILES := \
        built-in/op_proto/util/util.cc \
        built-in/op_proto/util/error_util.cc \
        built-in/op_proto/util/axis_util.cc \
        built-in/op_proto/selection_ops.cc \
        built-in/op_proto/elewise_calculation_ops.cc \
        built-in/op_proto/nonlinear_fuc_ops.cc \
        built-in/op_proto/nn_calculation_ops.cc \
        built-in/op_proto/nn_training_ops.cc \
        built-in/op_proto/nn_pooling_ops.cc \
        built-in/op_proto/pad_ops.cc \
        built-in/op_proto/quantize_ops.cc \
        built-in/op_proto/transformation_ops.cc \
        built-in/op_proto/util/array_ops_shape_fns.cc \
        built-in/op_proto/util/candidate_sampling_shape_fns.cc \
        built-in/op_proto/util/common_shape_fns.cc \
        built-in/op_proto/util/linalg_ops_shape_fns.cc \
        built-in/op_proto/util/images_ops_shape_fns.cc \
        built-in/op_proto/util/lookup_ops_shape_fns.cc \
        built-in/op_proto/util/nn_shape_fns.cc \
        built-in/op_proto/util/random_ops_shape_fns.cc \
        built-in/op_proto/util/ragged_conversion_ops_shape_fns.cc \
        built-in/op_proto/util/resource_variable_ops_shape_fns.cc \
        built-in/op_proto/util/transfer_shape_according_to_format.cc \
        built-in/op_proto/no_op.cc \
        built-in/op_proto/functional_ops.cc \
        built-in/op_proto/array_ops.cc \
        built-in/op_proto/control_flow_ops.cc \
        built-in/op_proto/aipp.cc \
        built-in/op_proto/audio_ops.cc \
        built-in/op_proto/batch_ops.cc \
        built-in/op_proto/bitwise_ops.cc \
        built-in/op_proto/boosted_trees_ops.cc \
        built-in/op_proto/candidate_sampling_ops.cc \
        built-in/op_proto/data_flow_ops.cc \
        built-in/op_proto/image_ops.cc \
        built-in/op_proto/linalg_ops.cc \
        built-in/op_proto/lookup_ops.cc \
        built-in/op_proto/math_ops.cc \
        built-in/op_proto/nn_ops.cc \
        built-in/op_proto/random_ops.cc \
        built-in/op_proto/set_ops.cc \
        built-in/op_proto/sparse_ops.cc \
        built-in/op_proto/state_ops.cc \
        built-in/op_proto/string_ops.cc \
        built-in/op_proto/matrix_calculation_ops.cc \
        built-in/op_proto/nn_batch_norm_ops.cc \
        built-in/op_proto/nn_norm_ops.cc \
        built-in/op_proto/nn_other_ops.cc \
        built-in/op_proto/reduce_ops.cc \
        built-in/op_proto/split_combination_ops.cc \
        built-in/op_proto/reduce_ops.cc \
        built-in/op_proto/nn_norm_ops.cc \
        built-in/op_proto/npu_loss_scale_ops.cc \
        built-in/op_proto/hcom_ops.cc \
	built-in/op_proto/hvd_ops.cc \
        built-in/op_proto/logging_ops.cc \
        built-in/op_proto/outfeed_ops.cc \
        built-in/op_proto/stateless_random_ops.cc \
        built-in/op_proto/roipooling_ops.cc \
        built-in/op_proto/rnn.cc \
        built-in/op_proto/fsrdetectionoutput_ops.cc \
        built-in/op_proto/ssddetectionoutput_ops.cc \
        built-in/op_proto/reduction_ops.cc \
        built-in/op_proto/copy_ops.cc \
        built-in/op_proto/psroipooling_ops.cc \
        built-in/op_proto/ragged_conversion_ops.cc \
        built-in/op_proto/stateful_random_ops.cc \
        built-in/op_proto/resource_variable_ops.cc \
        built-in/op_proto/ctc_ops.cc \
        built-in/op_proto/ragged_array_ops.cc \
        built-in/op_proto/ragged_math_ops.cc \
        built-in/op_proto/sdca_ops.cc \
        built-in/op_proto/nn_detect_ops.cc \
        built-in/op_proto/spectral_ops.cc \
        built-in/op_proto/condtake_ops.cc \
        built-in/op_proto/warp_perspective_ops.cc \
        built-in/op_proto/format_transfer_fractal_z.cc \
        built-in/op_proto/internal_ops.cc \
        built-in/op_proto/ifmr.cc \
        built-in/op_proto/wts_arq.cc \
        built-in/op_proto/acts_ulq.cc \
        built-in/op_proto/acts_ulq_input_grad.cc \
        built-in/op_proto/act_ulq_clamp_max_grad.cc \
        built-in/op_proto/act_ulq_clamp_min_grad.cc \
        built-in/op_proto/vector_search.cc \


LOCAL_C_INCLUDES := \
        proto/insert_op.proto \
        $(LOCAL_PATH)/../../inc/external/graph \
        $(LOCAL_PATH) \
        $(LOCAL_PATH)/../../common \
        $(LOCAL_PATH)/../../metadef \
        $(LOCAL_PATH)/../../metadef/inc \
        $(LOCAL_PATH)/../../metadef/inc/graph \
        $(LOCAL_PATH)/../../metadef/inc/external \
        $(LOCAL_PATH)/../../metadef/inc/external/graph \
        $(LOCAL_PATH)/../../graphengine/inc \
        $(LOCAL_PATH)/../../graphengine/ge \
        $(LOCAL_PATH)/../../graphengine/inc/external \
        $(LOCAL_PATH)/../../graphengine/inc/framework \
        $(LOCAL_PATH)/../../inc \
        $(LOCAL_PATH)/../../inc/cce \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/common/inc \
        $(LOCAL_PATH)/built-in/op_proto/inc \
        $(LOCAL_PATH)/built-in/op_proto/util \
        $(LOCAL_PATH)/built-in/framework/tf_plugin/util \
        $(LOCAL_PATH)/../../third_party/json/include \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../libc_sec/include \
        $(PLUGIN_C_INCLUDES)

LOCAL_SHARED_LIBRARIES := \
        libgraph \
        libslog  \
        libascend_protobuf \
        liberror_manager \
        libc_sec \
        libregister \

include $(BUILD_HOST_SHARED_LIBRARY)


# liboptiling.so
include $(CLEAR_VARS)

LOCAL_MODULE := liboptiling

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif

LOCAL_SRC_FILES := \
        built-in/fusion_pass/common/fp16_t.cc \
        built-in/op_proto/util/error_util.cc \
        built-in/op_tiling/auto_tiling.cc \
        built-in/op_tiling/unsorted_segment_sum.cc \
        built-in/op_tiling/unsorted_segment.cc \
        built-in/op_tiling/gather_nd.cc \
        built-in/op_tiling/gather_v2.cc \
        built-in/op_tiling/gather.cc \
        built-in/op_tiling/scatter_non_aliasing_add.cc \
        built-in/op_tiling/scatter_nd.cc \
        built-in/op_tiling/scatter_add.cc \
        built-in/op_tiling/scatter_sub.cc \
        built-in/op_tiling/scatter_update.cc \
        built-in/op_tiling/scatter_max.cc \
        built-in/op_tiling/scatter_min.cc \
        built-in/op_tiling/scatter_mul.cc \
        built-in/op_tiling/scatter_div.cc \
        built-in/op_tiling/scatter_nd_add.cc \
        built-in/op_tiling/scatter_nd_sub.cc \
        built-in/op_tiling/scatter_nd_update.cc \
        built-in/op_tiling/eletwise.cc \
        built-in/op_tiling/reduce_tiling.cc \
        built-in/op_tiling/sparse_apply_ftrl_d.cc \
        built-in/op_tiling/deconvolution.cc \
        built-in/op_tiling/dynamic_atomic_addr_clean.cc \
        built-in/op_tiling/sparse_apply_proximal_adagrad_d.cc \
        built-in/op_tiling/cube_tiling.cc \
        built-in/op_tiling/conv2d.cc \
        built-in/op_tiling/conv3d.cc \
        built-in/op_tiling/conv3d_backprop_input.cc \
        built-in/op_tiling/conv3d_transpose.cc \
        built-in/op_tiling/avg_pool.cc \
        built-in/op_tiling/avg_pool3d_grad.cc \
        built-in/op_tiling/conv2d_backprop_input.cc \
        built-in/op_tiling/conv2d_backprop_filter.cc \
        built-in/op_tiling/conv2d_transpose.cc \
        built-in/op_tiling/concat.cc \
        built-in/op_tiling/strided_slice.cc \
        built-in/op_tiling/slice.cc \
        built-in/op_tiling/transpose_d.cc \
        built-in/op_tiling/trans_data.cc \
	built-in/op_tiling/trans_data_positive_source_ntc_100.cc \
        built-in/op_tiling/unpack.cc \
        built-in/op_tiling/pad_common.cc \
        built-in/op_tiling/pad_d.cc \
        built-in/op_tiling/aipp.cc \
        built-in/op_tiling/split_d.cc \
        built-in/op_tiling/split_v.cc \
        built-in/op_tiling/select.cc \
        built-in/op_tiling/bias_add.cc \
        built-in/op_tiling/bias_add_grad.cc \
        built-in/op_tiling/assign.cc \
        built-in/op_tiling/avg_pool_grad.cc \
        built-in/op_tiling/strided_slice_grad.cc \
        built-in/op_tiling/fill.cc \
        built-in/op_tiling/tile_d.cc \
        built-in/op_tiling/tile.cc \
        built-in/op_tiling/max_pool.cc \
        built-in/op_tiling/arg_max_v2.cc \
        built-in/op_tiling/arg_max_with_value.cc \
        built-in/op_tiling/batch_to_space_nd.cc \
        built-in/op_tiling/space_to_batch_nd.cc \
        built-in/op_tiling/top_k.cc \
        built-in/op_tiling/gemm.cc \
        built-in/op_tiling/lock.cc \
        built-in/op_tiling/flatten.cc \
        built-in/op_tiling/nll_loss_grad.cc \
        built-in/op_tiling/dynamic_gru_cell_grad.cc \

LOCAL_C_INCLUDES := \
        $(LOCAL_PATH)/built-in/op_proto/util \
        $(LOCAL_PATH)/../../common \
        $(LOCAL_PATH)/common/inc \
        $(LOCAL_PATH)/../../metadef \
        $(LOCAL_PATH)/../../metadef/inc \
        $(LOCAL_PATH)/../../metadef/inc/external \
        $(LOCAL_PATH)/../../metadef/inc/external/graph \
        $(LOCAL_PATH)/../../graphengine/inc \
        $(LOCAL_PATH)/../../graphengine/inc/external \
        $(LOCAL_PATH)/../../graphengine/inc/framework \
        $(LOCAL_PATH)/../../inc \
        $(LOCAL_PATH)/../../inc/external \
        $(LOCAL_PATH)/../../libc_sec/include \
        $(LOCAL_PATH)/../../third_party/protobuf/include \
        $(LOCAL_PATH)/../../third_party/json/include \

LOCAL_SHARED_LIBRARIES := \
        libgraph \
        libslog  \
        libc_sec \
        libregister \
        liberror_manager \


include $(BUILD_HOST_SHARED_LIBRARY)

# tbe plugin for all caffe
include $(CLEAR_VARS)

LOCAL_MODULE := libops_all_caffe_plugin

#LOCAL_PATH := $(call my-dir)

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif

LOCAL_PLUGIN_DIRS_PATH := $(LOCAL_PATH)/built-in/framework/caffe_plugin

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc


LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_SRC_FILES := \
        built-in/op_proto/util/error_util.cc \
        $(LOCAL_PLUGIN_SRC_TARGET_FILES) \

#$(warning $(LOCAL_SRC_FILES))

LOCAL_C_INCLUDES := \
        $(PLUGIN_C_INCLUDES)

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations

LOCAL_SHARED_LIBRARIES := \
        libslog  \
        $(PLUGIN_SHARED_LIBS) \
        liberror_manager \

include $(BUILD_HOST_SHARED_LIBRARY)

####################################
# tbe plugin for all onnx
include $(CLEAR_VARS)

LOCAL_MODULE := libops_all_onnx_plugin

LOCAL_CFLAGS += -O2 -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0 -Dgoogle=ascend_private
endif

LOCAL_PLUGIN_DIRS_PATH := $(LOCAL_PATH)/built-in/framework/onnx_plugin

LOCAL_SRC_FILES_SUFFIX := %.cpp %.cc

LOCAL_PLUGIN_SRC_ALL_FILES := $(foreach src_path,$(LOCAL_PLUGIN_DIRS_PATH), $(shell find "$(src_path)" -type f) )
LOCAL_PLUGIN_SRC_ALL_FILES := $(LOCAL_PLUGIN_SRC_ALL_FILES:$(LOCAL_PLUGIN_DIRS_PATH)/./%=$(LOCAL_PLUGIN_DIRS_PATH)%)
LOCAL_PLUGIN_SRC_TARGET_FILES := $(filter $(LOCAL_SRC_FILES_SUFFIX),$(LOCAL_PLUGIN_SRC_ALL_FILES))
LOCAL_PLUGIN_SRC_TARGET_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES:$(LOCAL_PATH)/%=%)

LOCAL_SRC_FILES := $(LOCAL_PLUGIN_SRC_TARGET_FILES)

LOCAL_C_INCLUDES := \
        $(PLUGIN_C_INCLUDES)

LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations

LOCAL_SHARED_LIBRARIES := \
        libslog  \
        $(PLUGIN_SHARED_LIBS)  \
        liberror_manager \

include $(BUILD_HOST_SHARED_LIBRARY)

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend910-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend910
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend920-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend920
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend310-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend310
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend610-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend610
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################
########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend615-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend615
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-ascend710-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) ascend710
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-hi3796cv300es-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) hi3796cv300es
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-hi3796cv300cs-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) hi3796cv300cs
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aic-sd3403-ops-info.json

include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_tbe_info.sh $(product) sd3403
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)

#########################################

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := vector_core_tbe_ops_info.json

include $(LOCAL_PATH)/compile_vector_core_tbe_info_release.mk

#########################################

#########################################

# add source code for mini1951 (Interim plan)
include $(CLEAR_VARS)
LOCAL_MODULE := ops_impl
LOCAL_SRC_FILES := built-in/tbe/impl
LOCAL_MODULE_CLASS := FOLDER
LOCAL_INSTALLED_PATH :=  $(HOST_OUT_ROOT)/ops/op_impl/built-in/ai_core/tbe/impl
include $(BUILD_HOST_PREBUILT)

include $(CLEAR_VARS)
LOCAL_MODULE := ops_fusion_rules
LOCAL_SRC_FILES := built-in/fusion_rules
LOCAL_MODULE_CLASS := FOLDER
LOCAL_INSTALLED_PATH :=  $(HOST_OUT_ROOT)/ops/fusion_rules/built-in
include $(BUILD_HOST_PREBUILT)

########################################

include $(CLEAR_VARS)

LOCAL_MODULE := aicpu_kernel.json

include $(LOCAL_PATH)/compile_aicpu_info_release.mk

#########################################
