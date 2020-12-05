include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/aicpu/scripts/compile_tf_info.sh $@ $(product)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
